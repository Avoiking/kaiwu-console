"""测量后加载的 ctx 校验回归（2026-10-16，Ornith 疑案翻转后的真 bug 修复）。

病景（console 日志实证，见 Memorix obs:33 rev3）：自动测量的最后一试是
45056，掉了 3 层（31/34）但服务器没被卸掉；测量结论 32768 写盘后，
_start_inner 的"已在服务同一模型"短路只看模型名，不校验上下文 ⇒ 直接
返回 True，serve 的是 45056 掉层慢速档（21 t/s），而 cur_ctx 停在最后一次
尝试的 45056 ⇒ toast 报"上下文 44K"、卡片显示"上次测到 32K"，UI 自相矛盾。

修法：短路条件加 self._cur_ctx in (0, ctx)——跑着的服务上下文必须是
目标值才认；_cur_ctx == 0（非本实例拉起/从未加载）无从校验，维持放行。

本测试不起真 llama-server：probe/_stop_server_locked/Popen 全部打桩，
config 读写进内存。覆盖三个场景 + 重载日志文案。
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.config as app_config
from app import kaiwu_manager
from app.kaiwu_manager import KaiwuManager

PASS = FAIL = 0


def ok(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}")


# ---- 内存假磁盘：真实 config 的拷贝（imported_models/llama 路径等照旧） ----
STORE = dict(app_config.load_config())
kaiwu_manager.load_config = lambda: dict(STORE)


def _save(c):
    STORE.clear()
    STORE.update(c)


kaiwu_manager.save_config = _save


class FakeProc:
    pid = 999999
    returncode = None
    stdout = []

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


_real_popen = kaiwu_manager.subprocess.Popen
kaiwu_manager.subprocess.Popen = lambda *a, **k: FakeProc()

MODEL = "Ornith-1.5-9B"

# 目标上下文必须**自给自足**：干净 clone（CI、没有 data/config.json）里
# config 里根本没有这个模型，_target_ctx 会算出 0，断言就没意义了。
# 所以这里显式钉一个目标值 + 一个假 GGUF 路径，测试只关心
# 「服务器上下文 vs 目标值」的判定逻辑，不依赖本机登记状态。
TARGET_CTX = 32768
STORE.setdefault("ctx_measured", {})[MODEL] = TARGET_CTX
STORE.setdefault("ctx_measured_env", {})[MODEL] = {}
STORE.setdefault("ctx_auto", True)
STORE.setdefault("imported_models", {})[MODEL] = str(
    Path(__file__).resolve().parent.parent / "models" / MODEL / f"{MODEL}-Q4_K_M.gguf")


def make_manager(cur_ctx):
    mgr = KaiwuManager()
    mgr._cur_ctx = cur_ctx
    mgr.probe = lambda cfg=None, timeout=1.0: (True, [MODEL])   # 服务器"在跑"
    stops = []
    mgr._stop_server_locked = lambda cfg, port_cleanup=True: (stops.append(1) or True)
    mgr._should_measure = lambda cfg, model: False              # 隔离指纹/nvidia-smi
    mgr._resolve_gguf = lambda cfg, model: str(
        cfg.get("imported_models", {}).get(model) or "")
    return mgr, stops


target = KaiwuManager._target_ctx(STORE, MODEL)
print(f"\n（config 里 {MODEL} 的目标上下文 = {target}）")
ok(target == TARGET_CTX, "目标上下文来自测试自带的配置，不依赖本机登记状态")

print("\n[1] 病景：服务器停在测量失败档 45056，目标 32768 → 必须卸掉重载")
mgr, stops = make_manager(cur_ctx=45056)
ok(mgr.start(MODEL, dict(STORE), timeout=5) is True, "start 返回 True")
ok(bool(stops), "触发了卸载（旧行为会短路返回，不卸载）")
ok(mgr.cur_ctx() == target, f"cur_ctx 回到目标值（实际 {mgr.cur_ctx()}）")
ok(mgr.state() == ("ready", MODEL), "状态就绪")
logs = mgr.log_text()
ok("≠ 目标" in logs, f"重载日志说明了原因：{next((l for l in logs.splitlines() if '重载' in l), '')!r}")

print("\n[2] 对照：服务器上下文==目标 → 保持秒短路，不重载")
mgr, stops = make_manager(cur_ctx=target)
ok(mgr.start(MODEL, dict(STORE), timeout=5) is True, "start 返回 True")
ok(not stops, "没有卸载（快速路径不受影响）")
ok(mgr.cur_ctx() == target, "cur_ctx 不变")

print("\n[3] 无从校验：_cur_ctx=0（非本实例拉起）→ 维持放行")
mgr, stops = make_manager(cur_ctx=0)
ok(mgr.start(MODEL, dict(STORE), timeout=5) is True, "start 返回 True")
ok(not stops, "不卸载（无法证明上下文不符，保守放行）")

kaiwu_manager.subprocess.Popen = _real_popen
print(f"\n===== {PASS} 通过 / {FAIL} 失败 =====")
sys.exit(1 if FAIL else 0)
