"""队列合并 / 积压处理调度器（v2.3）

设计要点（对齐方案文档 v2.3）：
- 只在"当前批次（in-flight）处理中"时拦截后续批次进 pending；
- 推送决策三分支（都在 in-flight 完成时执行，串行）：
    分支① 软合并：pending 消息总数 <= 软合并上限 且 <= 合并消息数上限 且估 token <= 上限 -> 提前合并（不等超时）
    分支② 超时合并：pending 攒批时间 >= max_merge_seconds -> 合并（无论消息数，超限拆批）；
                   =0 时恒成立（不攒批，当前批次完成即全量合并，拆批由各上限控制）
    分支③ 独立推送：都不满足 -> 只推第一个批次（1:1），其余留 pending 等下一轮
- 用"事件配对"判定 in-flight 完成（0 延迟，无 release_delay）：
    ON_LLM_RESPONSE 无 tool_calls = 最后一步（文本收尾）-> 标记 _final_marked
    ON_STEP_RESULT（消息发送后触发）同 event_id 且已标记 -> 执行推送决策
- 自拦截防护：自己推送的（合并/重放）批次在 on_batch_message 直接放行，防死循环
- 积压媒体限制（media_preprocess_enabled + media_preprocess_max_batches）：
    含媒体批次在 pending 已满上限时直接放行独立处理，避免媒体无限积压 + VLM/STT 重复预处理
- 调试日志开关（debug_log_enabled）：开启后打印放行/拦截/三分支/拆批等状态，便于排查
- 合并批次必须沿用原 KiraIMMessage 引用（并行识图 _pir_images 依赖，绝不克隆）
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from core.plugin import logger
from core.chat.message_utils import KiraMessageBatchEvent
from core.chat.message_elements import Image, Sticker, Record
from core.provider import LLMResponse


@dataclass
class PendingBatch:
    """待推送批次（记录进入 pending 的时刻，用于超时合并判定）"""
    arrival_ts: float
    batch: KiraMessageBatchEvent


class BatchMergeScheduler:
    """队列合并调度器：作为 mixin 组件挂在聊天插件上，不引用插件私有状态。"""

    def __init__(self, ctx, plugin_cfg: dict, bot_cfg: dict):
        self.ctx = ctx
        sec = plugin_cfg.get("section_queue_merge", {})
        self.enabled = sec.get("enabled", True)
        self.max_merge_seconds = float(sec.get("max_merge_seconds", 0))
        self.max_merge_batches_limit = int(sec.get("max_merge_batches_limit", 0))
        self.max_merge_messages = int(sec.get("max_merge_messages", -1))
        self.max_merge_est_tokens = int(sec.get("max_merge_est_tokens", 0))
        self.token_est_ratio = float(sec.get("token_est_ratio", 2.0))
        self.short_merge_max_messages = int(sec.get("short_merge_max_messages", -1))
        self.media_preprocess_enabled = sec.get("media_preprocess_enabled", True)
        self.media_preprocess_max_batches = int(sec.get("media_preprocess_max_batches", 0))
        self.debug_log_enabled = sec.get("debug_log_enabled", False)

        # -1 自动解析
        buffer_cap = int(bot_cfg.get("max_buffer_messages", 5))
        recv_unmentioned = plugin_cfg.get("receive_unmentioned", False)
        unmentioned_cap = int(plugin_cfg.get("max_unmentioned_messages", 5)) if recv_unmentioned else 0
        if self.max_merge_messages == -1:
            # 自动 = (未提及消息缓冲上限 + 最大缓冲消息数) × 3；未提及关闭则不计前者
            self.max_merge_messages = (unmentioned_cap + buffer_cap) * 3
        if self.short_merge_max_messages == -1:
            # 软合并消息数上限默认继承 bot 的 max_buffer_messages
            self.short_merge_max_messages = buffer_cap

        # per-sid 状态
        self._inflight: dict[str, str] = {}          # sid -> event_id（当前正在处理的批次）
        self._final_marked: set[str] = set()         # 该 sid 的 in-flight 批次已进入最后一步
        self._pending: dict[str, list[PendingBatch]] = {}   # sid -> 待推送队列
        self._lock = asyncio.Lock()
        self._merge_task: Optional[asyncio.Task] = None

    # ================= 调试日志 =================

    def _log(self, sid: str, msg: str):
        """debug_log_enabled 开启时打印队列合并状态日志（info 级别，便于排查）。"""
        if self.debug_log_enabled:
            logger.info(f"[QueueMerge] {sid} {msg}")

    # ================= 钩子（由宿主插件的 @on.xxx 转发调用） =================

    async def on_batch_message(self, event: KiraMessageBatchEvent, *_):
        """ON_IM_BATCH_MESSAGE：主入口，放行 or 拦截入队。"""
        if not self.enabled:
            return
        sid = event.session.sid
        async with self._lock:
            if self._inflight.get(sid) == event.event_id:
                # 自己推送的（合并/重放）批次：in-flight 就是它自己，直接放行，
                # 防止"推送 -> 到达 -> 拦截自己 -> 超时再推 -> 再拦截"的死循环
                return
            if sid in self._inflight or self._pending.get(sid):
                # 积压媒体限制：pending 中已积压的含媒体批次达到上限时，
                # 新到的含媒体批次直接放行独立处理（媒体及时识别，不无限积压/重复 VLM）
                if (self.media_preprocess_enabled and self.media_preprocess_max_batches > 0
                        and self._has_media(PendingBatch(time.time(), event))
                        and self._count_media_batches(self._pending.get(sid, [])) >= self.media_preprocess_max_batches):
                    self._log(sid, f"媒体批次超积压上限({self.media_preprocess_max_batches})，直接放行独立处理 {event.event_id}")
                    return
                # 已有批次处理中 / 已有积压 -> 拦截进 pending
                self._pending.setdefault(sid, []).append(PendingBatch(time.time(), event))
                event.stop()
                pend_n = len(self._pending[sid])
                self._log(sid, f"拦截批次 {event.event_id} 进 pending（pending={pend_n}）")
                self._ensure_task_locked()
            else:
                # 空闲 -> 放行
                self._inflight[sid] = event.event_id
                self._log(sid, f"放行批次 {event.event_id}")

    async def on_llm_response(self, event: KiraMessageBatchEvent, resp: LLMResponse, *_):
        """ON_LLM_RESPONSE：无 tool_calls = 该批次最后一步（文本收尾）-> 标记。"""
        if not self.enabled:
            return
        async with self._lock:
            sid = event.session.sid
            if self._inflight.get(sid) == event.event_id and not resp.tool_calls:
                self._final_marked.add(sid)
                self._log(sid, f"批次 {event.event_id} 进入最后一步（文本收尾）")

    async def on_step_result(self, event: KiraMessageBatchEvent, *_):
        """ON_STEP_RESULT：最后一步消息已发送完（事实 #9）-> 执行推送决策（0 延迟）。"""
        if not self.enabled:
            return
        sid = event.session.sid
        need_push = False
        async with self._lock:
            if self._inflight.get(sid) == event.event_id and sid in self._final_marked:
                need_push = True
        if need_push:
            await self._push_pending(sid)

    # ================= 推送决策（三分支，串行） =================

    async def _push_pending(self, sid: str):
        """锁内决策 + 状态更新，锁外 publish。并发调用时第二个 pop 空直接返回，安全。"""
        to_publish = None
        async with self._lock:
            to_publish = self._decide_and_apply_locked(sid)
        if to_publish is not None:
            n_msgs = len(to_publish.messages)
            self._log(sid, f"发布批次 {to_publish.event_id}（{n_msgs} 条消息，来自 {len(to_publish.extra.get('merged_from', []))} 个来源）")
            await self.ctx.event_bus.publish(to_publish)

    def _decide_and_apply_locked(self, sid: str) -> Optional[KiraMessageBatchEvent]:
        """三分支推送决策（须持有 _lock）：返回要发布的合并批次，状态已更新。"""
        pending = self._pending.pop(sid, [])
        self._inflight.pop(sid, None)
        self._final_marked.discard(sid)
        if not pending:
            return None

        total_msgs = sum(len(pb.batch.messages) for pb in pending)
        est_tokens = self._estimate_tokens(pending)

        # 分支① 软合并：小积压提前合并（消息数 <= 软合并上限 且 <= 合并消息数上限 且 token 不超）
        if (total_msgs <= self.short_merge_max_messages
                and total_msgs <= self.max_merge_messages
                and (self.max_merge_est_tokens == 0 or est_tokens <= self.max_merge_est_tokens)):
            self._log(sid, f"软合并：{len(pending)}批次/{total_msgs}条（≤软合并上限{self.short_merge_max_messages}）")
            to_merge, rest = self._split_by_limits(pending)
        # 分支② 超时合并：攒批到点合并（无论消息多少，超限拆批）
        elif time.time() - pending[0].arrival_ts >= self.max_merge_seconds:
            waited = time.time() - pending[0].arrival_ts
            self._log(sid, f"超时合并：攒批 {waited:.1f}s ≥ {self.max_merge_seconds}s，{len(pending)}批次/{total_msgs}条")
            to_merge, rest = self._split_by_limits(pending)
        # 分支③ 独立推送：只推第一个批次（1:1），其余留 pending 等下一轮
        else:
            self._log(sid, f"独立推送：第1个批次（{len(pending[0].batch.messages)}条），其余 {len(pending) - 1} 个留 pending")
            to_merge, rest = [pending[0]], pending[1:]

        if rest:
            self._log(sid, f"拆批/留存：本次合 {len(to_merge)} 批次，{len(rest)} 批次留待下轮")

        self._pending[sid] = rest
        merged = self._build_merged_batch(to_merge)
        self._inflight[sid] = merged.event_id
        return merged

    # ================= 阈值防护 =================

    def _split_by_limits(self, pending: list[PendingBatch]):
        """按 批次数 / 媒体批次 / 消息数 / token 上限拆批，返回 (to_merge, rest)。"""
        to_merge: list[PendingBatch] = []
        rest: list[PendingBatch] = []
        total_msgs = 0
        total_tokens = 0
        media_batches = 0
        for pb in pending:
            msgs = len(pb.batch.messages)
            toks = self._estimate_tokens([pb])
            has_media = self._has_media(pb)
            if (self.max_merge_batches_limit and len(to_merge) >= self.max_merge_batches_limit):
                rest.append(pb)
                continue
            if (self.media_preprocess_enabled and self.media_preprocess_max_batches
                    and has_media and media_batches >= self.media_preprocess_max_batches):
                rest.append(pb)
                continue
            if self.max_merge_messages and total_msgs + msgs > self.max_merge_messages:
                rest.append(pb)
                continue
            if self.max_merge_est_tokens and total_tokens + toks > self.max_merge_est_tokens:
                rest.append(pb)
                continue
            to_merge.append(pb)
            total_msgs += msgs
            total_tokens += toks
            if has_media:
                media_batches += 1
        if not to_merge and pending:
            # 极端：第一个批次自身就超限 -> 仍合并它（1:1 语义，宁可超限不可丢/死循环）
            to_merge = [pending[0]]
            rest = pending[1:]
        return to_merge, rest

    def _estimate_tokens(self, batches: list[PendingBatch]) -> int:
        """粗略估算 token：文本字符数 / 估算系数。"""
        ratio = max(1, int(self.token_est_ratio))
        total = 0
        for pb in batches:
            for m in pb.batch.messages:
                msg_str = getattr(m, "message_str", None) or ""
                total += len(msg_str) // ratio
        return total

    def _has_media(self, pb: PendingBatch) -> bool:
        for m in pb.batch.messages:
            for elem in getattr(m, "chain", []) or []:
                if isinstance(elem, (Image, Sticker, Record)):
                    return True
        return False

    def _count_media_batches(self, pending: list[PendingBatch]) -> int:
        """统计 pending 中含媒体元素的批次数（积压媒体限制用）。"""
        return sum(1 for pb in pending if self._has_media(pb))

    # ================= 批次构造 =================

    def _build_merged_batch(self, batches: list[PendingBatch]) -> KiraMessageBatchEvent:
        """合并批次构造：必须沿用原 KiraIMMessage 引用（绝不克隆），属性取最后一批。"""
        msgs = []
        for pb in batches:
            msgs.extend(pb.batch.messages)
        last = batches[-1].batch
        merged = KiraMessageBatchEvent(
            message_types=last.message_types,
            timestamp=int(time.time()),
            adapter=last.adapter,
            session=last.session,
            messages=msgs,
            extra={"merged_from": [b.batch.event_id for b in batches]},
        )
        return merged

    # ================= 兜底 tick（仅 in-flight 卡死） =================

    def _ensure_task_locked(self):
        if self._merge_task is None or self._merge_task.done():
            self._merge_task = asyncio.create_task(self._tick_loop())

    async def _tick_loop(self):
        try:
            while True:
                await asyncio.sleep(0.5)
                await self._tick()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("[QueueMerge] tick loop error")

    async def _tick(self):
        to_publish = []
        async with self._lock:
            now = time.time()
            for sid in list(self._pending):
                pending = self._pending.get(sid)
                if not pending:
                    continue
                inflight = self._inflight.get(sid)
                if inflight and sid not in self._final_marked:
                    # in-flight 仍在处理（未到最后一步）——正常等待，仅当攒批超时（卡死兜底）才强制推送
                    if self.max_merge_seconds > 0 and now - pending[0].arrival_ts < self.max_merge_seconds:
                        continue
                    if self.max_merge_seconds > 0:
                        self._log(sid, f"超时兜底：in-flight 疑似卡死，强制推送 pending")
                    else:
                        # max_merge_seconds=0：不攒批，但仍由事件配对（当前批次完成）驱动，
                        # 避免 tick 在 in-flight 未完成时并发强制推送（0 语义 = 当前批次一完成立即合并推送）
                        continue
                merged = self._decide_and_apply_locked(sid)
                if merged is not None:
                    to_publish.append(merged)
        for merged in to_publish:
            n_msgs = len(merged.messages)
            self._log(merged.session.sid, f"发布批次 {merged.event_id}（{n_msgs} 条）")
            await self.ctx.event_bus.publish(merged)

    # ================= 生命周期 =================

    async def shutdown(self):
        """terminate 时调用：尽力重发 pending（不合并，逐个 1:1），取消 tick。可重入。"""
        task = None
        to_publish = []
        async with self._lock:
            task = self._merge_task
            self._merge_task = None
            for sid, pend in self._pending.items():
                for pb in pend:
                    to_publish.append(pb.batch)
                    self._log(sid, f"shutdown 重发 pending 批次 {pb.batch.event_id}")
            self._pending.clear()
            self._inflight.clear()
            self._final_marked.clear()
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        for batch in to_publish:
            try:
                await self.ctx.event_bus.publish(batch)
            except Exception:
                logger.exception("[QueueMerge] shutdown republish failed")
