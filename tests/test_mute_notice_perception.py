"""禁言感知回归测试：被禁言/解除禁言的 notice 必须能进 LLM（不能被入口过滤器吃掉）。

背景（2026-10-14 线上）：
  适配器构造 notice 事件时 sender.user_id = notice.user_id；而「bot 被禁言」这条
  notice 的 user_id 正是 bot 自己 ⇒ message.self_id == sender.user_id。
  宿主 handle_msg 第一道过滤器「丢弃机器人自己的消息」据此 event.discard()，
  于是 [System …禁言了你N秒] 永远进不了 LLM —— bot 事后声称「没收到任何通知」。

本测试用【真框架 + 真加载器 + 真插件代码】跑，事件与线上日志同构。
用法：
  python3 test_mute_notice_perception.py --core2x /path/to/2.x --core30 /path/to/3.0
  （也可只给其中一个；缺省读环境变量 KIRA_CORE_2X / KIRA_CORE_30）
"""
import asyncio
import os
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import real_loader_test as RLT  # noqa: E402  复用其 ctx 桩 / swap_core

PLUGIN_DIR = str(_HERE.parent)          # 插件根目录（tests/ 的上一层）
RLT.PLUGIN_DIR = PLUGIN_DIR             # 兼容其内部引用

BOT_ID = "3463604015"
OPERATOR = "3696040731"
GROUP = "115985242"
RESULTS = []


def check(gen, name, cond, detail=""):
    RESULTS.append((gen, name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] [{gen}] {name}" + (f" :: {detail}" if detail and not cond else ""),
          flush=True)


def ban_notice_chain(kind):
    """复刻 QQ增强(qx) 对 group_ban 的处理结果：给消息链追加 [System …]。"""
    from core.chat import MessageChain
    ch = MessageChain()
    if kind == "ban":
        ch.text(f"[System 用户{OPERATOR}(阿洛娜Arona（bot管理层）) 禁言了你120秒]")
    elif kind == "lift_ban":
        ch.text(f"[System 你之前被禁言了，用户{OPERATOR}(阿洛娜Arona（bot管理层）)解除了你的禁言]")
    return ch


def make_event(adapter, kind):
    from core.chat import Group, KiraIMMessage, MessageChain, User
    from core.chat.message_utils import KiraMessageEvent

    notice = kind in ("ban", "lift_ban", "empty_notice")
    if kind in ("ban", "lift_ban"):
        chain = ban_notice_chain(kind)
        raw = {"post_type": "notice", "notice_type": "group_ban",
               "sub_type": "ban" if kind == "ban" else "lift_ban",
               "self_id": BOT_ID, "user_id": BOT_ID, "operator_id": OPERATOR,
               "group_id": GROUP, "duration": 120 if kind == "ban" else 0, "time": 1786000000}
        mentioned, sender_id, nickname = True, BOT_ID, "Shana"
    elif kind == "empty_notice":
        chain, raw = MessageChain(), {"post_type": "notice", "notice_type": "notify",
                                      "sub_type": "poke", "self_id": BOT_ID, "user_id": BOT_ID,
                                      "target_id": "769690776", "group_id": GROUP, "time": 1786000000}
        mentioned, sender_id, nickname = False, BOT_ID, "Shana"
    elif kind == "self_echo":
        chain = MessageChain()
        chain.text("我自己刚发的一条群消息")
        raw, notice, mentioned, sender_id, nickname = None, False, False, BOT_ID, "Shana"
    elif kind == "user_msg":
        chain = MessageChain()
        chain.text("一分钟后我再找你，你必须接出下句")
        raw, notice, mentioned, sender_id, nickname = None, False, True, "769690776", "周武"
    else:
        raise SystemExit(f"unknown kind {kind}")

    msg = KiraIMMessage(timestamp=1786000000, message_id="None" if notice else "1764283028",
                        group=Group(group_id=GROUP, group_name="new Bot()"),
                        sender=User(user_id=str(sender_id), nickname=nickname),
                        is_notice=notice, is_mentioned=mentioned, self_id=BOT_ID,
                        chain=chain, raw_message=raw)
    return KiraMessageEvent(message_types=["gm"], timestamp=1786000000, message=msg, adapter=adapter)


async def load(gen, core):
    RLT.swap_core(core)
    from core.plugin import PluginContext, PluginManager
    ctx = RLT.build_ctx(gen, PluginContext)
    mgr = PluginManager(ctx)
    ctx.plugin_mgr = mgr
    pid = RLT.json_pid(PLUGIN_DIR)
    await mgr.load_plugin_from_dir(Path(PLUGIN_DIR), auto_install=False)
    inst = mgr.plugin_instances.get(pid) or mgr.get_plugin_inst(pid)
    # 隔离外部环境：确保不是被休眠/骚扰/评分门槛拦下（与本次审计无关的路径）
    try:
        inst.enhance.dormant.in_dormant = lambda *a, **k: False
        inst.enhance.harass.is_blocked = lambda *a, **k: False
        inst.enhance.harass.is_ignored = lambda *a, **k: False
    except Exception:
        pass
    return mgr, inst


async def run_gen(gen, core):
    mgr, inst = await load(gen, core)
    if inst is None:
        check(gen, "插件实例可用", False)
        return
    from core.adapter.adapter_info import AdapterInfo
    adapter = AdapterInfo(enabled=True, adapter_id="qq", name="qq", platform="QQ")

    async def probe(kind):
        ev = make_event(adapter, kind)
        try:
            await inst.handle_msg(ev)
        except Exception:
            # 后续路径缺桩属预期；只要没在这之前被丢弃即可
            pass
        return ev._process_strategy, getattr(ev, "_is_forced", False)

    # 1) 被禁言通知 —— 必须保留
    strat, _ = await probe("ban")
    check(gen, "★被禁言通知进入 LLM（不被当作自己的消息丢弃）", strat != "discard", f"策略={strat}")
    # 2) 解除禁言通知 —— 必须保留
    strat, _ = await probe("lift_ban")
    check(gen, "★解除禁言通知进入 LLM", strat != "discard", f"策略={strat}")
    # 3) 真·自己的消息回声 —— 必须仍然丢弃（安全设计）
    strat, _ = await probe("self_echo")
    check(gen, "自己消息回声仍然丢弃（安全不回归）", strat == "discard", f"策略={strat}")
    # 4) 空通知事件 —— 必须仍然丢弃（安全设计）
    strat, _ = await probe("empty_notice")
    check(gen, "空通知事件仍然丢弃（安全不回归）", strat == "discard", f"策略={strat}")
    # 5) 普通用户消息 —— 必须保留
    strat, _ = await probe("user_msg")
    check(gen, "普通用户消息不受影响", strat != "discard", f"策略={strat}")

    # 6) 同类路径 A：休眠期起夜未命中（抑制触发）
    #    通知必须仍然进上下文（只摘掉提及），而普通用户消息仍按原设计丢弃。
    async def probe_suppressed(kind, mode):
        ev = make_event(adapter, kind)
        enh = inst.enhance
        old_force = getattr(enh, "force_suppress", False)
        old_ratio = getattr(enh.presence, "ratio", None)
        old_score = getattr(enh.presence, "score", None)
        old_tr = getattr(enh.presence, "target_ratio", None)
        old_st = getattr(enh.presence, "score_threshold", None)
        old_in = enh.dormant.in_dormant
        old_awake = enh.dormant.is_awake
        old_tw = getattr(enh.dormant, "try_wake", None)
        try:
            if mode == "dormant":
                enh.dormant.in_dormant = lambda *a, **k: True
                enh.dormant.is_awake = lambda *a, **k: False
                enh.dormant.try_wake = lambda *a, **k: False
            else:  # force_suppress：占比超标 + 评分不足
                enh.force_suppress = True
                enh.presence.ratio = lambda *a, **k: 1.0
                enh.presence.score = lambda *a, **k: 0.0
                enh.presence.target_ratio = 0.0
                enh.presence.score_threshold = 1.0
            try:
                await inst.handle_msg(ev)
            except Exception:
                pass
            return ev
        finally:
            enh.dormant.in_dormant = old_in
            enh.dormant.is_awake = old_awake
            if old_tw is not None:
                enh.dormant.try_wake = old_tw
            enh.force_suppress = old_force
            if old_ratio is not None:
                enh.presence.ratio = old_ratio
            if old_score is not None:
                enh.presence.score = old_score
            if old_tr is not None:
                enh.presence.target_ratio = old_tr
            if old_st is not None:
                enh.presence.score_threshold = old_st

    for mode, label in (("dormant", "休眠抑制"), ("force", "存在感超额抑制")):
        ev = await probe_suppressed("ban", mode)
        check(gen, f"★{label}下禁言通知仍进上下文（通知不参与抑制引擎）",
              ev._process_strategy != "discard", f"策略={ev._process_strategy}")
        check(gen, f"{label}下禁言通知未被改写（提及态保留，交由 qx 决策）",
              bool(getattr(ev.message, "is_mentioned", False)) is True,
              f"提及={ev.message.is_mentioned}")
        ev = await probe_suppressed("user_msg", mode)
        check(gen, f"{label}下普通用户消息仍按原设计抑制（安全不回归）",
              ev._process_strategy == "discard", f"策略={ev._process_strategy}")

    try:
        await inst.terminate()
    except Exception:
        pass


CORE30 = os.environ.get("KIRA_CORE_30") or os.environ.get("KIRA_CORE") or "/var/minis/shared/kira30rc2"
CORE2X = os.environ.get("KIRA_CORE_2X") or "/var/minis/shared/kira_fw"


def main():
    targets = []
    core2x, core30 = CORE2X, CORE30
    if "--core2x" in sys.argv:
        core2x = sys.argv[sys.argv.index("--core2x") + 1]
    if "--core30" in sys.argv:
        core30 = sys.argv[sys.argv.index("--core30") + 1]
    if os.path.isdir(core2x):
        targets.append(("2.x", core2x))
    if os.path.isdir(core30):
        targets.append(("3.0", core30))
    if not targets:
        print("SKIPPED：未提供框架路径（用 --core2x/--core30 或环境变量 KIRA_CORE_2X/KIRA_CORE_30 指定），"
              "本套件需要真框架才能跑。")
        sys.exit(0)
    print(f"\n===== 禁言感知回归：{Path(PLUGIN_DIR).name} =====")
    for gen, core in targets:
        try:
            asyncio.run(run_gen(gen, core))
        except Exception:
            check(gen, "套件异常", False, traceback.format_exc()[-600:])
    fails = [r for r in RESULTS if not r[2]]
    print(f"\n===== 汇总：{len(RESULTS) - len(fails)}/{len(RESULTS)} 通过 =====")
    if fails:
        print("失败项：")
        for gen, name, _ in fails:
            print(f"  - [{gen}] {name}")
        sys.exit(1)
    print("MUTE NOTICE PERCEPTION TEST PASSED")


if __name__ == "__main__":
    main()
