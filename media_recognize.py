"""并行媒体识别模块（v2.3.1）—— 图片 VLM + 音频 STT 并行预处理

设计要点（对齐方案文档 KiraAI并行媒体识别模块对齐方案.md v1.0）：
- 三阶段标识符架构（照搬并行识图插件的成熟模式，扩展音频）：
    stage1 (ON_IM_MESSAGE)     先把嵌套 Forward 就地拍平（借鉴并行识图插件 _flatten_forwards，
                               防核心渲染过滤嵌套 Forward 丢内容），再把 Image/Sticker/Record
                               替换为标识符 [Image #id: ] / [Record #id: ]，阻止框架 format_to_text
                               串行识别；原始元素暂存到消息动态属性 _pir_media
    stage2 (ON_IM_BATCH_MESSAGE) 收集批次内全部暂存媒体，同一 gather 混合并行识别
                               （图片 VLM / 音频 STT 各自三层限流：批次级 batch_sem →
                               会话级 _session_img_sems/_session_aud_sems → 全局级
                               _global_img_sem/_global_aud_sem，固定顺序无死锁），
                               填充 message_str 与 chain —— 积压批次在拦截前识别完成 = 真预处理
    stage3 (ON_LLM_REQUEST)    历史/当前残留空标识符兜底（缓存命中→填；有原媒体→现场识别；否则 (已过期)）
- 缓存：复用框架 image_desc_cache 表（图片 md5→描述；音频 to_base64 md5→transcript），零 DB 改动
- VLM 描述词：跟随 WebUI 配置 bot_config.capabilities.image_recognition.desc_prompt
  （对齐框架 message_format_to_text 行为；未配置时用 locale.lang 语言默认 prompt）
- 兼容：compat_mode=auto 检测 parallel_image_reader 插件——装了则图片归它（优先级 99 先处理）、
        本模块只做音频（它不碰 Record）；不装则全权接管图片+音频
- 原生多模态：native 模式运行时实时检测（_native_mode()，非 __init__ 快照）——
  用户在 WebUI 切换 bot_config.capabilities.image_recognition.mode 立即生效，
  无需重启 Kira / 重载插件（配置走内存缓存，微秒级，零延迟）
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
from core.utils.common_utils import get_default_vlm_prompt, speech_to_text

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
        # 三层并发限制（VLM / STT 各自独立），按 批次级 → 会话级 → 全局级 依次获取：
        #   ① 批次级（max_parallel_images / max_parallel_audios）：单个批次内同时识别的最大数，
        #      防"一批 10 张图一次全轰出去"的突发；每个批次使用独立临时信号量
        #   ② 会话级（vlm/stt_max_parallel_per_session）：单个会话累积的最大并行数
        #   ③ 全局级（vlm/stt_max_parallel_global）：所有会话合计的最大并行数
        self.max_parallel_images = int(sec.get("max_parallel_images", 3))
        self.max_parallel_audios = int(sec.get("max_parallel_audios", 3))
        self.vlm_max_parallel_per_session = int(sec.get("vlm_max_parallel_per_session", 15))
        self.vlm_max_parallel_global = int(sec.get("vlm_max_parallel_global", 40))
        self.stt_max_parallel_per_session = int(sec.get("stt_max_parallel_per_session", 15))
        self.stt_max_parallel_global = int(sec.get("stt_max_parallel_global", 40))
        self.media_timeout = float(sec.get("media_timeout", 60.0))
        self.compat_mode = sec.get("compat_mode", "auto")
        self.quality_enabled = sec.get("quality_enabled", False)
        self.quality_value = int(sec.get("quality_value", 85))

        self._global_img_sem = asyncio.Semaphore(max(1, self.vlm_max_parallel_global))
        self._global_aud_sem = asyncio.Semaphore(max(1, self.stt_max_parallel_global))
        # 每会话信号量（惰性创建，热重载后自动重建）
        self._session_img_sems: dict[str, asyncio.Semaphore] = {}
        self._session_aud_sems: dict[str, asyncio.Semaphore] = {}

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

        # 原生多模态模式（KiraAI v2.31.0+）：bot_config.capabilities.image_recognition.mode == "native"
        # 时，图片由框架原生多模态直接传给模型（官方压缩 + kira_image_ref 持久化引用），
        # 本模块只做音频 STT，stage1 不再替换 Image/Sticker —— 否则 _build_native_content
        # 遍历 chain 找不到图片元素，原生多模态内容为空（模型收不到图），且 stage2 仍会
        # 调用 VLM 描述，与 native 模式"省 VLM token 直传图片"的初衷冲突。
        # 注意：非唤醒消息的图片仍由宿主 handle_msg 按"非唤醒不识别"策略替换为 [图片] 占位，
        # 只有唤醒消息的图片会保留并走原生多模态 —— 与 z/s 版省 token 设计一致。
        # 模式检测不做 __init__ 快照，改由 _native_mode() 每次事件实时读取（见下）：
        # 用户在 WebUI 直接切换 mode 而不重启/重载时，快照会过时——切到 native 后 stage1 仍
        # 替换图片（原生多模态收不到图）、切回 vlm_description 后图片无人识别（VLM 被跳过）。
        # 实时读配置走内存缓存（微秒级），无卡顿无延迟，WebUI 保存后立即生效。

        # 动态属性挂载名（沿用并行识图插件协议语义）
        self._media_attr = "_pir_media"
        # 当前回合暂存原媒体的 id 索引（stage3 现场识别用）。
        # 按 sid 分层：多会话并发处理时互不串扰
        self._round_media: dict[str, dict[str, dict]] = {}

    # ================= 调试日志 =================

    def _log(self, msg: str):
        logger.debug(f"[MediaRecognize] {msg}")

    def _pir_active(self) -> bool:
        """运行时实时检测并行识图插件（PIR）是否已加载。

        compat_mode=auto 时图片归 PIR（本模块只做音频）。不能像旧实现那样在
        __init__ 里做一次性快照：插件可热重载/启停，快照会过时——PIR 中途卸载后
        图片无人处理、中途加载后双重处理。
        """
        if self.compat_mode != "auto":
            return False
        try:
            pm = getattr(self.ctx, "plugin_mgr", None)
            if pm is not None:
                return pm.get_plugin_inst("parallel_image_reader") is not None
        except Exception:
            pass
        return False

    def _native_mode(self) -> bool:
        """运行时实时检测原生多模态模式（KiraAI v2.31.0+）。

        与 _pir_active 同理，不做 __init__ 一次性快照：用户可能在 WebUI 直接切换
        bot_config.capabilities.image_recognition.mode 而不重启 Kira / 重载插件，
        快照会过时。每次事件实时读取配置（框架配置走内存缓存，微秒级，
        无卡顿延迟），WebUI 保存后立即生效。
        """
        try:
            if hasattr(self.ctx, "config") and self.ctx.config is not None:
                mode = self.ctx.config.get_config(
                    "bot_config.capabilities.image_recognition.mode", "vlm_description"
                )
                return str(mode or "").lower() == "native"
        except Exception:
            pass
        return False

    # ================= 三级并发限流（批次级 + 每会话 + 全局） =================

    def _session_sem(self, sems: dict, sid: str, limit: int) -> asyncio.Semaphore:
        """惰性获取/创建某会话的信号量。"""
        sem = sems.get(sid)
        if sem is None:
            sem = asyncio.Semaphore(max(1, limit))
            sems[sid] = sem
        return sem

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
                # 合并而非覆盖：并行识图插件（PIR）可能已先写入 Image 索引，
                # 直接覆盖会让它 stage2/stage3 拿不到图片（图片标识符永远空）
                existing = getattr(event.message, self._media_attr, None) or {}
                setattr(event.message, self._media_attr, {**existing, **media})
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
                # 并行识图插件已加载且 auto 模式：图片归它，本模块不碰。
                # 运行时实时检测（不是 __init__ 快照），PIR 热重载/启停后自动生效
                if self._pir_active():
                    continue
                # 原生多模态模式（KiraAI v2.31.0+）：图片保留在 chain 中，
                # 由框架 _build_native_content 收集并直传模型（官方压缩 + 持久化引用）。
                # 本模块不替换、不识别图片，只做音频 STT。
                if self._native_mode():
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
        """媒体 → 标识符 Text；缓存命中填内容、miss 空标识符 + 暂存原元素。

        _done 标记语义：标识符已含最终内容（缓存描述）或已识别过（含失败），
        stage2 重发（队列合并重放）时跳过——每条消息的每个媒体最多识别一次，
        避免同一条消息被反复 VLM/STT（限流/429 风暴源头）。
        """
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
        media[short_id] = {"md5": md5, "elem": elem, "type": mtype, "_done": bool(desc)}
        if desc:
            # 缓存命中：直接带 file_path（to_path 幂等，_temp_path 已缓存不重复下载），
            # 对齐原版 message_format_to_text 的 [Image desc, file_path: xxx] 格式
            p = await self._media_path(elem)
            if p:
                return Text(f"[{mtype} #{short_id}: {desc}, file_path: {p}]")
        return Text(f"[{mtype} #{short_id}: {desc}]")

    async def _media_path(self, elem) -> Optional[str]:
        """对齐原版 message_format_to_text：to_path 落盘后转 data/ 相对路径。

        原版（core/message_manager.py Image 分支）：to_path() → relative_to(data_dir)
        → "data/xxx"，失败降级绝对路径。本模块 stage1 把媒体替换为标识符绕过了
        原版渲染，这里补回 file_path，让 LLM 能拿到本地路径做图生图/上传等。
        """
        try:
            from pathlib import Path
            from core.utils.path_utils import get_data_path
            path = Path(await elem.to_path())
            data_dir = get_data_path()
            try:
                rel = path.relative_to(data_dir)
                return f"data/{rel}"
            except ValueError:
                return str(path)
        except Exception:
            return None

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

            # 当前回合原媒体索引（stage3 用）；按 sid 分层防多会话并发串扰，
            # 同一 sid 的并发批次用 setdefault+update 合并，避免后到批次清掉先到批次
            sess_sid = event.session.sid
            self._round_media.setdefault(sess_sid, {})
            for _, media in tasks:
                for short_id, info in media.items():
                    self._round_media[sess_sid][short_id] = info
            # 防无界增长：最多保留 128 个 sid 的索引，超出清最旧
            if len(self._round_media) > 128:
                for old_sid in list(self._round_media)[: len(self._round_media) - 64]:
                    self._round_media.pop(old_sid, None)

            # 只识别未处理（_done=False）的媒体：缓存命中（stage1 已填描述）或
            # 已识别过（成功/失败）的跳过——队列合并重发同一批消息时不会重复 VLM/STT
            pending_tasks = [
                (message, {k: v for k, v in media.items() if not v.get("_done")})
                for message, media in tasks
            ]
            pending_tasks = [(m, md) for m, md in pending_tasks if md]
            # 原生多模态模式：图片已由框架直传模型，stage2 只做音频 STT
            if self._native_mode():
                pending_tasks = [
                    (m, {k: v for k, v in md.items() if v.get("type") != "Image"})
                    for m, md in pending_tasks
                ]
                pending_tasks = [(m, md) for m, md in pending_tasks if md]

            # 混合并行：图片 VLM 与 音频 STT 同一 gather，各自限流互不阻塞。
            # 批次级信号量：每批次临时创建，限制本批次内同时识别的数量（突发保护）
            batch_img_sem = asyncio.Semaphore(max(1, self.max_parallel_images))
            batch_aud_sem = asyncio.Semaphore(max(1, self.max_parallel_audios))
            results: dict[str, str] = {}
            coros = []
            for _, media in pending_tasks:
                for short_id, info in media.items():
                    if info["type"] == "Image":
                        coros.append(self._describe_one(sess_sid, short_id, info, results, batch_sem=batch_img_sem))
                    else:
                        coros.append(self._transcribe_one(sess_sid, short_id, info, results, batch_sem=batch_aud_sem))
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

    async def _describe_one(self, sess_sid: str, media_id: str, info: dict, results: dict,
                            batch_sem: Optional[asyncio.Semaphore] = None):
        md5 = info["md5"]
        cached = await self._cache_get(md5) if md5 else None
        if cached:
            info["_done"] = True
            results[media_id] = cached
            return
        try:
            sess_sem = self._session_sem(self._session_img_sems, sess_sid, self.vlm_max_parallel_per_session)
            # 三层限流：批次级 → 会话级 → 全局级（固定获取顺序，无死锁）
            if batch_sem is not None:
                async with batch_sem, sess_sem, self._global_img_sem:
                    desc = await asyncio.wait_for(self._describe_image(info["elem"]), self.media_timeout)
            else:
                async with sess_sem, self._global_img_sem:
                    desc = await asyncio.wait_for(self._describe_image(info["elem"]), self.media_timeout)
            # 无论成功失败都标记已处理：同一条消息重发不再重复识别（防 429 风暴）
            info["_done"] = True
            if desc and self._is_valid_desc(desc):
                if md5:
                    await self._cache_set(md5, desc)
                results[media_id] = desc
            else:
                logger.warning(f"[MediaRecognize] image VLM returned empty/invalid desc id={media_id} md5={md5[:8] if md5 else 'n/a'}")
                results[media_id] = "(未识别)"
        except Exception as e:
            info["_done"] = True
            logger.warning(f"[MediaRecognize] image describe failed id={media_id}: {type(e).__name__}: {e}")
            results[media_id] = "(未识别)"

    async def _transcribe_one(self, sess_sid: str, media_id: str, info: dict, results: dict,
                              batch_sem: Optional[asyncio.Semaphore] = None):
        md5 = info["md5"]
        cached = await self._cache_get(md5) if md5 else None
        if cached:
            info["_done"] = True
            results[media_id] = cached
            return
        try:
            provider_mgr = getattr(self.ctx, "provider_mgr", None)
            stt_client = provider_mgr.get_default_stt() if provider_mgr is not None else None
            if stt_client is None:
                info["_done"] = True
                logger.warning(f"[MediaRecognize] STT client unavailable (no default STT model) id={media_id}")
                results[media_id] = "(未识别)"
                return
            sess_sem = self._session_sem(self._session_aud_sems, sess_sid, self.stt_max_parallel_per_session)
            # 三层限流：批次级 → 会话级 → 全局级（固定获取顺序，无死锁）
            if batch_sem is not None:
                async with batch_sem, sess_sem, self._global_aud_sem:
                    text = await asyncio.wait_for(
                        speech_to_text(client=stt_client, record=info["elem"]), self.media_timeout)
            else:
                async with sess_sem, self._global_aud_sem:
                    text = await asyncio.wait_for(
                        speech_to_text(client=stt_client, record=info["elem"]), self.media_timeout)
            # 无论成功失败都标记已处理：同一条消息重发不再重复识别（防 429 风暴）
            info["_done"] = True
            if text and self._is_valid_desc(text):
                if md5:
                    await self._cache_set(md5, text)
                results[media_id] = text
            else:
                logger.warning(f"[MediaRecognize] STT returned empty/invalid text id={media_id}")
                results[media_id] = "(未识别)"
        except Exception as e:
            info["_done"] = True
            logger.warning(f"[MediaRecognize] STT failed id={media_id}: {type(e).__name__}: {e}")
            results[media_id] = "(未识别)"

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
            # 批次级限流同样作用于 stage3 兜底识别（一个 LLM 请求内的残留标识符 = 一个批次）
            batch_img_sem = asyncio.Semaphore(max(1, self.max_parallel_images))
            batch_aud_sem = asyncio.Semaphore(max(1, self.max_parallel_audios))
            coros = []
            # 只查本会话当前回合暂存的媒体（按 sid 分层，多会话不串扰）
            round_media = self._round_media.get(event.sid, {})
            for media_id in need:
                info = round_media.get(media_id)
                if info and not info.get("_done"):
                    # 有原媒体且未识别过 → 现场识别
                    if info["type"] == "Image":
                        # 原生多模态模式：图片不识别，直接标 (未识别) 占位
                        if self._native_mode():
                            results[media_id] = "(未识别)"
                            continue
                        coros.append(self._describe_one(event.sid, media_id, info, results, batch_sem=batch_img_sem))
                    else:
                        coros.append(self._transcribe_one(event.sid, media_id, info, results, batch_sem=batch_aud_sem))
                elif info:
                    # 已识别过但占位符仍空（异常路径）：直接标未识别，不重复撞模型
                    results[media_id] = "(未识别)"
                else:
                    results[media_id] = "(已过期)"
            if coros:
                await asyncio.gather(*coros, return_exceptions=True)
            for p in getattr(req, "user_prompt", []) or []:
                text = getattr(p, "content", "") or ""
                new_text = self._fill_text(text, results)
                if new_text != text:
                    p.content = new_text
        except Exception:
            logger.exception("[MediaRecognize] stage3 error")
        finally:
            # 无论正常/异常/提前 return 都清理本会话暂存媒体索引，防单 sid 无限累积（内存泄漏）。
            # stage2 的 setdefault+update 是同步原子块，pop 后新批次会重建，无并发风险
            self._round_media.pop(event.sid, None)

    # ================= 填充 =================

    def _fill_text(self, text: str, results: dict) -> str:
        for sid, desc in results.items():
            # 用 str.replace 而非 re.sub：replacement 是模板字符串，desc 含 \U/\x 等
            # 反斜杠序列（如 Windows 路径）会抛 bad escape；replace 无转义问题
            text = text.replace(f"[Image #{sid}: ]", f"[Image #{sid}: {desc}]")
            text = text.replace(f"[Record #{sid}: ]", f"[Record #{sid}: {desc}]")
        return text

    def _fill_chain(self, chain, results: dict):
        if chain is None:
            return
        for elem in chain:
            if isinstance(elem, Text):
                # 与 _fill_text 一致：全文 replace（不依赖 match 只匹配开头），
                # 避免 Text 前有前缀时 chain 漏填而 message_str 已填的不一致
                new_text = self._fill_text(elem.text or "", results)
                if new_text != elem.text:
                    elem.text = new_text
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
