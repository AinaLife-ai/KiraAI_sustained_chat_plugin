"""并行媒体识别模块（v2.3）—— 图片 VLM + 音频 STT 并行预处理

设计要点（对齐方案文档 KiraAI并行媒体识别模块对齐方案.md v1.0）：
- 三阶段标识符架构（照搬并行识图插件的成熟模式，扩展音频）：
    stage1 (ON_IM_MESSAGE)     先把嵌套 Forward 就地拍平（借鉴并行识图插件 _flatten_forwards，
                               防核心渲染过滤嵌套 Forward 丢内容），再把 Image/Sticker/Record
                               替换为标识符 [Image #id: ] / [Record #id: ]，阻止框架 format_to_text
                               串行识别；原始元素暂存到消息动态属性 _pir_media
    stage2 (ON_IM_BATCH_MESSAGE) 收集批次内全部暂存媒体，同一 gather 混合并行识别
                               （图片 VLM 走 _sem_img、音频 STT 走 _sem_aud，各自限流互不阻塞），
                               填充 message_str 与 chain —— 积压批次在拦截前识别完成 = 真预处理
    stage3 (ON_LLM_REQUEST)    历史/当前残留空标识符兜底（缓存命中→填；有原媒体→现场识别；否则 (已过期)）
- 缓存：复用框架 image_desc_cache 表（图片 md5→描述；音频 to_base64 md5→transcript），零 DB 改动
- VLM 描述词：跟随 WebUI 配置 bot_config.capabilities.image_recognition.desc_prompt
  （对齐框架 message_format_to_text 行为；未配置时用 locale.lang 语言默认 prompt）
- 兼容：compat_mode=auto 检测 parallel_image_reader 插件——装了则图片归它（优先级 99 先处理）、
        本模块只做音频（它不碰 Record）；不装则全权接管图片+音频
- 与 z/sustained"非唤醒不识别"兼容：im_message 钩子须定义在 handle_msg 之后（handle_msg 先替换
  非唤醒媒体为 [图片]/[语音] 占位，本模块后执行链上已无媒体 → 不识别非唤醒消息）
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import re
from io import BytesIO
from typing import Optional

from core.plugin import logger
from core.chat.message_utils import KiraMessageEvent, KiraMessageBatchEvent
from core.chat.message_elements import Text, Image, Sticker, Record, Reply, Forward
from core.provider import LLMRequest
from core.utils.common_utils import get_default_vlm_prompt

# 标识符匹配（内容三态：空 / 描述 / (未识别) (已过期)）
_IMAGE_RE = re.compile(r"\[Image #([^\]\s:]+): ([^\]]*)\]")
_RECORD_RE = re.compile(r"\[Record #([^\]\s:]+): ([^\]]*)\]")
_ALL_RE = re.compile(r"\[(?:Image|Record) #([^\]\s:]+): ([^\]]*)\]")


class ParallelMediaRecognizer:
    """并行媒体识别：作为 mixin 组件挂在聊天插件上，与 queue_merge 解耦。"""

    def __init__(self, ctx, plugin_cfg: dict, bot_cfg: dict):
        self.ctx = ctx
        sec = plugin_cfg.get("section_media_recognition", {})
        self.enabled = sec.get("enabled", True)
        self.max_parallel_images = int(sec.get("max_parallel_images", 3))
        self.max_parallel_audios = int(sec.get("max_parallel_audios", 3))
        self.media_timeout = float(sec.get("media_timeout", 60.0))
        self.compat_mode = sec.get("compat_mode", "auto")
        self.quality_enabled = sec.get("quality_enabled", False)
        self.quality_value = int(sec.get("quality_value", 85))

        self._sem_img = asyncio.Semaphore(self.max_parallel_images)
        self._sem_aud = asyncio.Semaphore(self.max_parallel_audios)

        # VLM 描述语言：读全局 locale.lang；未设置默认中文（对齐并行识图插件中文 DESC_PROMPT）。
        # 实际 prompt 优先取 WebUI 配置 desc_prompt（§_describe_image），此处 lang 仅作默认兜底
        self._vlm_lang = "zh"
        try:
            if hasattr(ctx, "config") and ctx.config is not None:
                cfg_lang = ctx.config.get_config("locale.lang")
                if cfg_lang:
                    self._vlm_lang = str(cfg_lang)
        except Exception:
            pass

        # 检测并行识图插件是否已加载（图片归它 / 本模块只做音频）
        self._pir_loaded = False
        try:
            pm = getattr(ctx, "plugin_mgr", None)
            if pm is not None:
                self._pir_loaded = pm.get_plugin_inst("parallel_image_reader") is not None
        except Exception:
            self._pir_loaded = False

        # 动态属性挂载名（沿用并行识图插件协议语义）
        self._media_attr = "_pir_media"
        # 当前回合暂存原媒体的 id 索引（stage3 现场识别用）
        self._round_media: dict[str, dict] = {}

    # ================= 调试日志 =================

    def _log(self, msg: str):
        logger.debug(f"[MediaRecognize] {msg}")

    # ================= stage1：拍平嵌套 Forward + 替换为标识符 =================

    # 递归遍历/拍平的深度上限：防恶意超深嵌套（Forward 层层套娃）触发 RecursionError。
    # 超深时安全降级——深层 Forward 保留原样，由核心过滤兜底（内容无痕省略但不崩溃）。
    _MAX_CHAIN_DEPTH = 64

    @staticmethod
    def _flatten_forwards(chain, stack=None, depth=0, max_depth=None):
        """就地拍平嵌套 Forward（借鉴并行识图插件 _flatten_forwards，main.py:234-285）。

        KiraAI 核心 message_format_to_text 渲染 Forward 时会过滤嵌套 Forward 元素
        （`[x for x in chain if not isinstance(x, Forward)]`，message_manager.py:371，防无限递归），
        导致嵌套转发的内容（含图片标识符）不进 message_str，LLM 看不到。stage1 先把嵌套
        Forward 展开为平铺元素，保证嵌套内容完整渲染。

        语义：depth=0 的顶层 Forward（消息本身是转发）保留壳；depth>0 的嵌套 Forward
        逐层展开为其子链内容。覆盖路径：Forward.chains 与 Reply.chain。防环：stack 记录
        当前展开路径上的 chain（id），环中子链保留 Forward 元素（核心过滤兜底）。
        深度上限 max_depth（默认 _MAX_CHAIN_DEPTH）：超限不展开（深层内容无痕省略）。
        """
        if max_depth is None:
            max_depth = ParallelMediaRecognizer._MAX_CHAIN_DEPTH
        if stack is None:
            stack = set()
        cid = id(chain)
        if cid in stack:
            return  # 环：同一展开路径上再次出现
        stack.add(cid)
        i = 0
        while i < len(chain):
            ele = chain[i]
            if isinstance(ele, Reply) and ele.chain is not None:
                if depth < max_depth:
                    ParallelMediaRecognizer._flatten_forwards(
                        ele.chain, stack, depth + 1, max_depth)
            elif isinstance(ele, Forward) and ele.chains:
                if depth < max_depth:
                    for c in ele.chains:
                        ParallelMediaRecognizer._flatten_forwards(
                            c, stack, depth + 1, max_depth)
                if depth > 0 and depth < max_depth:
                    # 嵌套 Forward：展开为其子链内容（平铺替换元素本身）
                    expanded = []
                    for c in ele.chains:
                        if id(c) in stack:
                            continue  # 环：跳过该子链（内容无痕省略）
                        expanded.extend(c)
                    if expanded:
                        chain[i:i + 1] = expanded
                        i += len(expanded) - 1
            i += 1
        stack.remove(cid)

    async def on_im_message(self, event: KiraMessageEvent, *_):
        """ON_IM_MESSAGE：先拍平嵌套 Forward（防核心渲染丢内容），再遍历替换媒体为标识符并暂存。"""
        if not self.enabled:
            return
        try:
            self._flatten_forwards(event.message.chain)
            media: dict[str, dict] = {}
            await self._walk_chain(event.message.chain, media, set())
            if media:
                setattr(event.message, self._media_attr, media)
        except Exception:
            logger.exception("[MediaRecognize] stage1 error")

    async def _walk_chain(self, chain, media: dict, visited: set):
        """递归遍历 chain（含 Reply.chain / Forward.chains，带环检测）。嵌套 Forward 已拍平。"""
        if chain is None:
            return
        cid = id(chain)
        if cid in visited:
            return
        visited.add(cid)
        for idx, elem in enumerate(chain):
            if isinstance(elem, Text):
                continue
            if isinstance(elem, (Image, Sticker)):
                # 并行识图插件已加载且 auto 模式：图片归它，本模块不碰
                if self._pir_loaded and self.compat_mode == "auto":
                    continue
                replaced = await self._replace_media(elem, "Image", media)
                if replaced is not None:
                    chain[idx] = replaced
            elif isinstance(elem, Record):
                replaced = await self._replace_media(elem, "Record", media)
                if replaced is not None:
                    chain[idx] = replaced
            elif isinstance(elem, Reply):
                await self._walk_chain(getattr(elem, "chain", None), media, visited)
            elif isinstance(elem, Forward):
                for sub in (getattr(elem, "chains", None) or []):
                    await self._walk_chain(sub, media, visited)

    async def _replace_media(self, elem, mtype: str, media: dict) -> Optional[Text]:
        """媒体 → 标识符 Text；缓存命中填内容、miss 空标识符 + 暂存原元素。"""
        try:
            if mtype == "Record":
                md5 = await self._record_md5(elem)
            else:
                md5 = await elem.hash_image()
        except Exception:
            md5 = None
        if md5:
            short_id = md5[:8]
            desc = await self._cache_get(md5) or ""
            if desc and not self._is_valid_desc(desc):
                desc = ""
        else:
            short_id = f"noid_{id(elem)}"
            desc = ""
        media[short_id] = {"md5": md5, "elem": elem, "type": mtype}
        return Text(f"[{mtype} #{short_id}: {desc}]")

    async def _record_md5(self, elem) -> Optional[str]:
        """音频指纹：to_base64 后取 md5（Record 无 hash_image）。"""
        try:
            b64 = await elem.to_base64()
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[1]
            return hashlib.md5(base64.b64decode(b64)).hexdigest()
        except Exception:
            return None

    # ================= stage2：并行识别 + 填充（核心） =================

    async def on_im_batch_message(self, event: KiraMessageBatchEvent, *_):
        """ON_IM_BATCH_MESSAGE：收集批次暂存媒体，VLM 与 STT 混合 gather 并行识别，填充。"""
        if not self.enabled:
            return
        try:
            tasks = []  # [(message, media)]
            for message in event.messages:
                media = getattr(message, self._media_attr, None)
                if media:
                    tasks.append((message, media))
            if not tasks:
                return

            # 当前回合原媒体索引（stage3 用）
            self._round_media = {}
            for _, media in tasks:
                for sid, info in media.items():
                    self._round_media[sid] = info

            # 混合并行：图片 VLM 与 音频 STT 同一 gather，各自限流互不阻塞
            results: dict[str, str] = {}
            coros = []
            for _, media in tasks:
                for sid, info in media.items():
                    if info["type"] == "Image":
                        coros.append(self._describe_one(sid, info, results))
                    else:
                        coros.append(self._transcribe_one(sid, info, results))
            await asyncio.gather(*coros, return_exceptions=True)

            # 填充 message_str 与 chain
            for message, media in tasks:
                hit = any(sid in results for sid in media)
                if hit:
                    if message.message_str:
                        message.message_str = self._fill_text(message.message_str, results)
                    self._fill_chain(message.chain, results)
        except Exception:
            logger.exception("[MediaRecognize] stage2 error")

    async def _describe_one(self, sid: str, info: dict, results: dict):
        md5 = info["md5"]
        cached = await self._cache_get(md5) if md5 else None
        if cached:
            results[sid] = cached
            return
        try:
            async with self._sem_img:
                desc = await asyncio.wait_for(self._describe_image(info["elem"]), self.media_timeout)
            if desc and self._is_valid_desc(desc):
                if md5:
                    await self._cache_set(md5, desc)
                results[sid] = desc
            else:
                logger.warning(f"[MediaRecognize] image VLM returned empty/invalid desc sid={sid} md5={md5[:8] if md5 else 'n/a'}")
                results[sid] = "(未识别)"
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"[MediaRecognize] image describe failed sid={sid}: {type(e).__name__}: {e}")
            results[sid] = "(未识别)"

    async def _transcribe_one(self, sid: str, info: dict, results: dict):
        md5 = info["md5"]
        cached = await self._cache_get(md5) if md5 else None
        if cached:
            results[sid] = cached
            return
        try:
            async with self._sem_aud:
                text = await asyncio.wait_for(
                    self.ctx.llm_api.speech_to_text(record=info["elem"]), self.media_timeout)
            if text and self._is_valid_desc(text):
                if md5:
                    await self._cache_set(md5, text)
                results[sid] = text
            else:
                logger.warning(f"[MediaRecognize] STT returned empty/invalid text sid={sid}")
                results[sid] = "(未识别)"
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"[MediaRecognize] STT failed sid={sid}: {type(e).__name__}: {e}")
            results[sid] = "(未识别)"

    async def _describe_image(self, elem) -> str:
        """图片 VLM：统一 to_data_url → vlm.chat 路径（对齐并行识图插件已验证路径）；
        to_data_url 失败时 fallback 直接 httpx 下载（带 UA + pixiv Referer，覆盖图床防盗链）；
        quality_enabled 时 JPEG 压缩。失败返回 ""（调用方降级为 (未识别) 并打日志）。"""
        try:
            vlm = self.ctx.provider_mgr.get_default_vlm()
            if vlm is None:
                logger.warning("[MediaRecognize] get_default_vlm() returned None")
                return ""
            data_url = None
            try:
                data_url = await elem.to_data_url()
            except Exception as e:
                logger.debug(f"[MediaRecognize] to_data_url failed ({type(e).__name__}), try direct download")
                data_url = await self._try_direct_download(elem)
            if not data_url:
                logger.warning(
                    f"[MediaRecognize] cannot fetch image data: "
                    f"file_type={getattr(elem, 'file_type', '?')} "
                    f"file={str(getattr(elem, 'file', ''))[:80]}"
                )
                return ""
            if self.quality_enabled:
                _, _, b64 = data_url.partition(",")
                if not b64:
                    logger.warning("[MediaRecognize] empty base64 after to_data_url")
                    return ""
                img = _open_image(base64.b64decode(b64))
                q = max(10, min(100, self.quality_value))
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=q)
                data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
            prompt = self._vlm_prompt()
            request = LLMRequest(messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                    {"type": "text", "text": prompt},
                ],
            }])
            resp = await vlm.chat(request)
            return (resp.text_response or "").strip() if resp else ""
        except Exception as e:
            logger.warning(f"[MediaRecognize] describe image failed: {type(e).__name__}: {e}")
            return ""

    def _vlm_prompt(self) -> str:
        """VLM 描述词：跟随 WebUI 配置 bot_config.capabilities.image_recognition.desc_prompt
        （对齐框架 message_format_to_text 行为）；未配置/为空时用 locale.lang 语言默认 prompt。"""
        try:
            caps = self.ctx.config.get_config("bot_config.capabilities.image_recognition", {})
            desc_prompt = (caps or {}).get("desc_prompt", "") or ""
            if desc_prompt.strip():
                return desc_prompt.strip()
        except Exception:
            pass
        return get_default_vlm_prompt(self._vlm_lang)

    async def _try_direct_download(self, elem) -> Optional[str]:
        """to_data_url 失败时：直接 httpx 下载图片（带 UA，pixiv 图床补 Referer），返回 data_url 或 None。"""
        url = getattr(elem, "file", None) or getattr(elem, "image", None)
        if not url or not str(url).startswith(("http://", "https://")):
            return None
        try:
            import httpx
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            }
            async with httpx.AsyncClient(follow_redirects=True, timeout=self.media_timeout) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code != 200:
                    # pixiv 图床防盗链：补 Referer 重试
                    resp = await client.get(url, headers={**headers, "Referer": "https://www.pixiv.net/"})
                if resp.status_code == 200 and resp.content:
                    return "data:image/jpeg;base64," + base64.b64encode(resp.content).decode()
        except Exception as e:
            logger.debug(f"[MediaRecognize] direct download failed: {type(e).__name__}: {e}")
        return None

    # ================= stage3：历史/残留标识符兜底 =================

    async def on_llm_request(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        """ON_LLM_REQUEST：扫描 req.user_prompt 残留空标识符：缓存命中填、有原媒体现场识别、否则 (已过期)。"""
        if not self.enabled:
            return
        try:
            need: dict[str, str] = {}  # sid -> 标识符类型
            for p in getattr(req, "user_prompt", []) or []:
                text = getattr(p, "content", "") or ""
                for m in _ALL_RE.finditer(text):
                    if not m.group(2).strip():
                        need[m.group(1)] = m.group(0)
            if not need:
                return
            results: dict[str, str] = {}
            coros = []
            for sid in need:
                info = self._round_media.get(sid)
                if info:
                    if info["type"] == "Image":
                        coros.append(self._describe_one(sid, info, results))
                    else:
                        coros.append(self._transcribe_one(sid, info, results))
                else:
                    results[sid] = "(已过期)"
            if coros:
                await asyncio.gather(*coros, return_exceptions=True)
            for p in getattr(req, "user_prompt", []) or []:
                text = getattr(p, "content", "") or ""
                new_text = self._fill_text(text, results)
                if new_text != text:
                    p.content = new_text
        except Exception:
            logger.exception("[MediaRecognize] stage3 error")

    # ================= 填充 =================

    def _fill_text(self, text: str, results: dict) -> str:
        for sid, desc in results.items():
            text = re.sub(rf"\[Image #{re.escape(sid)}: \]", f"[Image #{sid}: {desc}]", text)
            text = re.sub(rf"\[Record #{re.escape(sid)}: \]", f"[Record #{sid}: {desc}]", text)
        return text

    def _fill_chain(self, chain, results: dict):
        if chain is None:
            return
        for elem in chain:
            if isinstance(elem, Text):
                m = _ALL_RE.match(elem.text or "")
                if m and not m.group(2).strip() and m.group(1) in results:
                    prefix = "Image" if elem.text.startswith("[Image") else "Record"
                    elem.text = re.sub(
                        rf"\[(?:Image|Record) #{re.escape(m.group(1))}: \]",
                        f"[{prefix} #{m.group(1)}: {results[m.group(1)]}]",
                        elem.text,
                    )
            elif isinstance(elem, Reply):
                self._fill_chain(getattr(elem, "chain", None), results)
            elif isinstance(elem, Forward):
                for sub in (getattr(elem, "chains", None) or []):
                    self._fill_chain(sub, results)

    # ================= 缓存（复用 image_desc_cache 表） =================

    async def _cache_get(self, md5: str) -> Optional[str]:
        try:
            row = await self.ctx.db.get_image_desc_cache(md5)
            return row["description"] if row else None
        except Exception:
            return None

    async def _cache_set(self, md5: str, text: str):
        if not md5 or not text:
            return
        try:
            await self.ctx.db.add_image_desc_cache(md5, text, count=1, last_seen=0)
        except Exception:
            pass

    # ================= 校验 =================

    @staticmethod
    def _is_valid_desc(desc: str) -> bool:
        if not desc or not desc.strip():
            return False
        if "\x00" in desc:
            return False
        if "<!--PIR:" in desc:
            return False
        if "[Image #" in desc or "[Record #" in desc:
            return False  # 防嵌套标识符注入缓存并扩散
        return True


def _open_image(data: bytes):
    from PIL import Image as PILImage
    return PILImage.open(BytesIO(data)).convert("RGB")
