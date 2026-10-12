"""S/Z 版：3.0 rc1 兼容 —— 异步化能力接口 + SQLite 会话持久化（#360）。

rc1（v3.0.0-rc.1, 8a0b812）相对 alpha.4 的破坏性变化：
  · `session_manager.get_effective_capabilities(...)` 变 async（插件同步调用会拿到
    coroutine → isinstance(dict) 判假 → 静默退化全局配置 + RuntimeWarning 刷屏）；
  · `fetch_memory` / `write_memory` 移除 ⇒ 会话记忆走 `get_session()` /
    `update_session(content=)`（异步）。

本套件断言：
  C1 同步读点在 rc1（异步接口）下：不抛、不产 RuntimeWarning、退化全局配置
  C2 异步预热后，同步读点拿到**会话级覆盖**（缓存 TTL 内）
  C3 旧世代（同步接口）行为不变：直读会话覆盖、不走缓存通道
  C4 _session_memory_chunks 双世代：fetch_memory 路径 + get_session 路径
  C5 _write_memory_chunks 双世代：write_memory 路径 + update_session 路径
  C6 _dm_state_text 在 rc1 记忆接口下产出文本（S 版）

Run: python3 tests/test_rc1_compat.py [<plugin_dir> ...]
"""
import asyncio
import importlib.util
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
STUB = HERE / "_stub"
sys.path.insert(0, str(STUB))

results = []

def check(label, cond, detail=""):
    results.append((label, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + label
          + ((" — " + str(detail)) if (detail and not cond) else ""))

def load_mods(plugin_dir: Path):
    for m in ("main", "queue_merge", "media_recognize", "chat_enhance", "jev"):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(plugin_dir))
    spec = importlib.util.spec_from_file_location(f"main_rc1_{plugin_dir.name}",
                                                  plugin_dir / "main.py")
    main = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = main
    spec.loader.exec_module(main)
    return main, sys.modules["media_recognize"]

class _Cfg:
    def __init__(self, caps=None):
        self._caps = caps or {}

    def get_config(self, key, default=None):
        if key == "bot_config.capabilities":
            return self._caps
        return default

    def __getitem__(self, key):
        return {"agent": {"max_tool_loop": 2}, "bot": {}} if key == "bot_config" else {}

class _Rc1SessionMgr:
    """rc1 形态：get_effective_capabilities 是 async；记忆走 get_session/update_session。"""
    def __init__(self, caps=None):
        self._caps = caps or {}
        self.content = []
        self.writes = []

    async def get_effective_capabilities(self, sid, global_caps):
        return dict(self._caps or global_caps)

    async def get_session(self, sid, ensure_session=False, include_content=True):
        return SimpleNamespace(content=self.content)

    async def update_session(self, sid, content=None, **kw):
        self.writes.append((sid, content))
        if content is not None:
            self.content = content
        return SimpleNamespace()

class _LegacySessionMgr:
    """2.x 形态：同步 fetch_memory/write_memory + 同步 get_effective_capabilities。"""
    def __init__(self, caps=None):
        self._caps = caps or {}
        self.memory = []
        self.writes = []

    def get_effective_capabilities(self, sid, global_caps):
        return dict(self._caps or global_caps)

    def fetch_memory(self, sid):
        return self.memory

    def write_memory(self, sid, chunks):
        self.writes.append((sid, chunks))
        self.memory = chunks

def _recognizer(mr, sm):
    ctx = SimpleNamespace(config=_Cfg(), session_mgr=sm,
                          provider_mgr=None)
    return mr.ParallelMediaRecognizer(ctx, {}, {})

# ---------------------------------------------------------------- C1/C2/C3
async def c1_c2_rc1_async_caps(mr):
    rec = _recognizer(mr, _Rc1SessionMgr(caps={"image_recognition": {"mode": "native"}}))
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        # 同步读点（预热前）：退化全局（本桩全局为空 → mode None → vlm 语义 False）
        r0 = rec._native_mode("s1")
        await asyncio.sleep(0)
        bad = [w for w in ws if "never awaited" in str(w.message)]
        # 异步预热后再读：拿到会话级 native
        await rec._warm_caps_cache("s1")
        r1 = rec._native_mode("s1")
    return {
        "C1 rc1 同步读点不产 RuntimeWarning": len(bad) == 0,
        "C1 rc1 预热前退化全局（非 native）": r0 is False,
        "C2 预热后同步读点拿到会话覆盖（native=True）": r1 is True,
        "C2 缓存已登记": "s1" in rec._caps_cache,
    }

async def c3_legacy_sync_caps(mr):
    rec = _recognizer(mr, _LegacySessionMgr(caps={"image_recognition": {"mode": "native"}}))
    r = rec._native_mode("s1")
    return {
        "C3 旧世代同步直读会话覆盖": r is True,
        "C3 旧世代不走缓存通道": len(rec._caps_cache) == 0 and rec._caps_async_api is not True,
    }

# ---------------------------------------------------------------- C4/C5/C6
async def c4_c5_memory_dual(main_mod):
    plug_ctx = SimpleNamespace(config=_Cfg())
    # 2.x 形态
    leg = _LegacySessionMgr()
    plug = main_mod.DebouncePlugin(SimpleNamespace(config=_CfgMain(), session_mgr=leg,
                                                   message_processor=None), _s_cfg())
    got = await plug._session_memory_chunks("s1")
    ok1 = got == []
    await plug._write_memory_chunks("s1", [[{"role": "user", "content": "hi"}]])
    ok2 = leg.writes and leg.memory == [[{"role": "user", "content": "hi"}]]
    ok3 = await plug._session_memory_chunks("s1") == [[{"role": "user", "content": "hi"}]]
    # rc1 形态
    rc1 = _Rc1SessionMgr()
    plug2 = main_mod.DebouncePlugin(SimpleNamespace(config=_CfgMain(), session_mgr=rc1,
                                                    message_processor=None), _s_cfg())
    ok4 = await plug2._session_memory_chunks("s1") == []
    await plug2._write_memory_chunks("s1", [[{"role": "user", "content": "早"}]])
    ok5 = rc1.writes and rc1.content == [[{"role": "user", "content": "早"}]]
    ok6 = await plug2._session_memory_chunks("s1") == [[{"role": "user", "content": "早"}]]
    return {
        "C4 2.x 读空记忆": ok1, "C5 2.x write_memory 路径": ok2, "C4 2.x 读回": ok3,
        "C4 rc1 读空记忆": ok4, "C5 rc1 update_session 路径": ok5, "C4 rc1 读回": ok6,
    }

class _CfgMain:
    def get_config(self, key, default=None):
        return {"bot_config.agent.max_tool_loop": 2,
                "bot_config.bot.max_buffer_messages": 5}.get(key, default)

    def __getitem__(self, key):
        return {"agent": {"max_tool_loop": 2}, "bot": {"max_buffer_messages": 5}} \
            if key == "bot_config" else {}

def _s_cfg():
    return {"section_basic": {}, "section_group_sustain": {}, "section_dm_sustain": {}}

async def c6_dm_state_rc1(main_mod):
    if not hasattr(main_mod.DebouncePlugin, "_dm_state_text"):
        return {"C6 z 版无私聊 state（按设计跳过）": True}
    rc1 = _Rc1SessionMgr()
    rc1.content = [[{"role": "user", "content": "在吗"},
                    {"role": "assistant", "content": "在的"}]]
    plug = main_mod.DebouncePlugin(SimpleNamespace(config=_CfgMain(), session_mgr=rc1,
                                                   message_processor=None), _s_cfg())
    txt = await plug._dm_state_text("s1")
    return {"C6 rc1 记忆渲染出文本": "在吗" in txt and "在的" in txt,
            "C6 角色前缀正确": "user: 在吗" in txt and "bot: 在的" in txt}

async def run(plugin_dir: Path):
    main_mod, mr = load_mods(plugin_dir)
    for label, cond in (await c1_c2_rc1_async_caps(mr)).items():
        check(label, cond)
    for label, cond in (await c3_legacy_sync_caps(mr)).items():
        check(label, cond)
    if hasattr(main_mod.DebouncePlugin, "_session_memory_chunks"):
        for label, cond in (await c4_c5_memory_dual(main_mod)).items():
            check(label, cond)
        for label, cond in (await c6_dm_state_rc1(main_mod)).items():
            check(label, cond)
    else:
        print("  (z 版：记忆读写用例按设计跳过)")

def main():
    dirs = [Path(d).resolve() for d in (sys.argv[1:] or [".."])]
    for d in dirs:
        print("=" * 78)
        print(f"### {d.name}")
        asyncio.run(run(d))
    failed = [l for l, c in results if not c]
    print("=" * 78)
    if failed:
        print(f"{len(failed)} CHECK(S) FAILED")
        for l in failed:
            print("  FAIL", l)
        sys.exit(1)
    print("ALL TESTS PASSED")

if __name__ == "__main__":
    main()
