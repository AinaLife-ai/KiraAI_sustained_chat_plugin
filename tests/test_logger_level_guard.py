"""S/Z：日志级别守卫 —— 不得写 `logger.DEBUG`（线上 AttributeError，整条钩子中断）。

线上报错（2026-10-13 20:04，KiraAI v2.16.1）：

    chat_enhance.py:970   if _delta and logger.DEBUG:
    AttributeError: 'Logger' object has no attribute 'DEBUG'

`DEBUG` 是标准库 `logging` **模块**上的常量，logger **实例没有**这个属性。
后果不止"少一行日志"：异常从 `on_im_message` 里冒出去 ⇒ **整条钩子中断**，
它后面那段骚扰检测 / 休眠判定**全被跳过**（用户侧静默失效，只在日志里留一条 ERROR）。

断言：
  G1 静态：源码里不得出现 `xxx.<级别>`（把级别当属性用）—— 用 AST 扫，注释/文档串不算；
  G2 真 `logging.Logger`：DEBUG 开 ⇒ True；级别高于 DEBUG ⇒ False；
  G3 桩 logger：没有 `isEnabledFor` ⇒ 按 `level` 兜底；`isEnabledFor` 抛错 ⇒ False；
  G4 陌生对象 / None ⇒ False（**绝不抛异常**）；
  G5 端到端：用**真** `on_im_message` 驱动一条命中加分关键词的用户消息 ⇒
     **不抛异常**，且分数确实被补正（修前这一条直接复现线上报错）。

Run: python3 tests/test_logger_level_guard.py [<plugin_dir> ...]
"""
import ast
import asyncio
import importlib.util
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUB = HERE / "_stub"
sys.path.insert(0, str(STUB))

from core.chat import Group, KiraIMMessage, MessageChain, Session, User  # noqa: E402
from core.chat.message_elements import Text  # noqa: E402

results = []


def check(label, cond, detail=""):
    results.append((label, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + label
          + ((" — " + str(detail)) if (detail and not cond) else ""))


def load_plugin(plugin_dir: Path):
    for m in ("main", "queue_merge", "media_recognize", "chat_enhance"):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(plugin_dir))
    spec = importlib.util.spec_from_file_location(f"main_lg_{plugin_dir.name}", plugin_dir / "main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Cfg:
    def get_config(self, key, default=None):
        return {"bot_config.agent.max_tool_loop": 2,
                "bot_config.agent.tool_call_timeout": 60,
                "bot_config.bot.max_buffer_messages": 5,
                "bot_config.bot.max_message_interval": 30}.get(key, default)

    def __getitem__(self, key):
        return {}


class _Ctx:
    def __init__(self):
        self.config = _Cfg()

    def get_plugin_data_dir(self):
        return Path("/tmp")


class SEvent:
    """最小事件桩（与其它套件的 SEvent 同形）。"""

    def __init__(self, message, session):
        self.message = message
        self.session = session
        self.adapter = type("A", (), {"name": "qq", "platform": "qq"})()
        self.message_types = []
        self.is_mentioned = False
        self.is_notice = False

    @property
    def sid(self):
        return self.session.sid

    def is_group_message(self):
        return True


def make_event(text: str):
    m = KiraIMMessage(timestamp=time.time(), sender=User("9CD54739", "周武"),
                      group=Group("427674145", "测试群"), message_id="m1",
                      self_id="10000", chain=MessageChain([Text(text)]),
                      is_mentioned=False)
    return SEvent(m, Session(adapter_name="qq", session_type="gm", session_id="427674145"))


LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"}


def scan_level_attr(plugin_dir: Path):
    """AST 扫描：`xxx.<级别>`（把日志级别当属性用）。注释/文档字符串不会命中。"""
    bad = []
    for p in sorted(plugin_dir.glob("*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            bad.append(f"{p.name}: 语法错误 {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in LEVELS:
                try:
                    base = ast.unparse(node.value)
                except Exception:  # noqa: BLE001  (py<3.9)
                    base = getattr(node.value, "id", "") or ""
                if base.endswith("logging"):
                    # `logging.DEBUG` / `_logging.DEBUG`（模块别名）都是**合法**写法
                    continue
                bad.append(f"{p.name}:{node.lineno}: {base}.{node.attr}")
    return bad


async def run(plugin_dir: Path):
    mod = load_plugin(plugin_dir)
    ce = sys.modules["chat_enhance"]

    bad = scan_level_attr(plugin_dir)
    check("G1 源码里没有 `logger.<级别>`（把级别当属性用；AST 扫描）",
          not bad, "; ".join(bad))

    helper = getattr(ce, "_debug_enabled", None)     # 旧代码没有它 ⇒ 下面几条应报红
    lg = logging.getLogger("lg-guard-real-debug")
    lg.setLevel(logging.DEBUG)
    check("G2a 真 logging.Logger（DEBUG 级）⇒ True",
          callable(helper) and helper(lg) is True, f"helper={helper!r}")
    lg2 = logging.getLogger("lg-guard-real-warn")
    lg2.setLevel(logging.WARNING)
    check("G2b 真 logging.Logger（WARNING 级）⇒ False",
          callable(helper) and helper(lg2) is False)

    class _NoApi:
        level = logging.DEBUG

    check("G3a 没有 isEnabledFor ⇒ 按 level 兜底（DEBUG ⇒ True）",
          callable(helper) and helper(_NoApi()) is True)

    class _NoApiHigh:
        level = logging.ERROR

    check("G3b 没有 isEnabledFor 且级别高 ⇒ False", callable(helper) and helper(_NoApiHigh()) is False)

    class _Boom:
        def isEnabledFor(self, lv):
            raise RuntimeError("boom")

    check("G3c isEnabledFor 抛错 ⇒ 保守 False（绝不冒泡）", callable(helper) and helper(_Boom()) is False)

    class _Bare:
        pass

    check("G4 陌生对象 / None ⇒ False", callable(helper) and helper(_Bare()) is False
          and helper(None) is False)

    # G5 端到端
    plug = mod.DebouncePlugin(_Ctx(), {
        "section_basic": {}, "section_group_sustain": {}, "section_dm_sustain": {},
        "score_boost_words": ["谢谢"], "score_boost_values": [7],
    })
    eng = plug.enhance
    ev = make_event("谢谢哥！")
    pres = eng._get_presence(False)
    before = pres.score(ev.sid, time.time())
    err = None
    try:
        eng.on_im_message(ev)
    except Exception as exc:          # 修前就是这里抛 AttributeError
        err = f"{type(exc).__name__}: {exc}"
    check("G5a 真 on_im_message（命中加分词）**不得抛异常**", err is None, err)
    after = pres.score(ev.sid, time.time())
    check("G5b 关键词确实被计分（证明那段代码真的跑到了）", after > before, f"{before} → {after}")


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or [HERE.parent]
    for d in dirs:
        print(f"########## {d} ##########")
        asyncio.run(run(d))
    ok = all(c for _, c in results)
    print("\n" + ("ALL PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
