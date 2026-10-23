"""加载守卫测试（2026-10-16）：验证并发加载被明确拒绝而不是排队挂起。

覆盖三点：
  1. 异线程并发请求 → start() 毫秒级返回 False + busy_loading=True
     + last_error 指明"正在加载谁"；HTTP 层映射 409 {"busy": true}
  2. 同线程重入（measure_ctx → attempt → start 自己）→ 放行，不被误拦
  3. finally 解锁：即使 _start_inner 抛异常，"加载中"标志也不会永久卡住

不起真实 llama-server、不读显卡，纯逻辑 + HTTP 层验证。
"""
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app.config as app_config            # noqa: E402
from app import webui                      # noqa: E402
from app.kaiwu_manager import KaiwuManager   # noqa: E402

s = socket.socket()
s.bind(("127.0.0.1", 0))
free = s.getsockname()[1]
s.close()
_real_load = app_config.load_config
webui.load_config = lambda: {**_real_load(), "webui_port": free}

mgr = KaiwuManager()
ui = webui.WebUI(mgr)
port = ui.start()
assert port == free, (port, free)
time.sleep(0.3)

BASE = f"http://127.0.0.1:{free}"

def post_use(name):
    req = urllib.request.Request(f"{BASE}/api/use",
                                 data=json.dumps({"name": name}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

# ---------- 1. 异线程并发 → 拒绝 ----------
started, release = threading.Event(), threading.Event()
def holder():
    mgr._loading_tid = threading.get_ident()      # 模拟"线程1正在加载"
    mgr._loading_model = "HoldingModel"
    started.set()
    try:
        release.wait(10)
    finally:
        # 模拟真实加载结束：最外层 start() 的 finally 清理标志
        mgr._loading_tid = None
        mgr._loading_model = ""
t = threading.Thread(target=holder, daemon=True)
t.start()
assert started.wait(2), "holder 线程没起来"

t0 = time.time()
ok = mgr.start("AnotherModel", _real_load())       # 主线程（异线程）发起加载
dt = time.time() - t0
assert ok is False, "并发请求应被拒绝"
assert dt < 1.0, f"拒绝应毫秒级，实际 {dt:.2f}s"
assert mgr.busy_loading is True
assert "HoldingModel" in (mgr.last_error or ""), mgr.last_error
print(f"1a OK 管理器层拒绝：{dt*1000:.0f}ms，last_error={mgr.last_error!r}")

code, body = post_use("AnotherModel")              # 走 HTTP 层验证 409
assert code == 409, (code, body)
assert body.get("busy") is True and "HoldingModel" in body.get("error", ""), body
print(f"1b OK HTTP 层：409 busy=true error={body['error']!r}")

release.set(); t.join(2)                           # 放掉 holder，清场给后续用例
assert not mgr.busy_loading

# ---------- 2. 同线程重入 → 放行 ----------
# 真实重入路径：外层 start() 持标志 → _start_inner 内部（模拟 measure_ctx 的
# attempt()）在同一线程再调 start() → 应穿过守卫，且最终由外层 finally 解锁。
calls = []
def fake_inner(model, cfg=None, timeout=None, ctx_override=None, force_measure=False,
               moe_cpu=False, extra_args=None):
    if model == "OuterModel":
        inner = mgr.start("InnerModel", cfg, 5, ctx_override=4096)  # 同线程重入
        calls.append(("inner-returned", inner))
        return True
    calls.append(("inner-leaf-reached", model))
    return False
orig_inner = mgr._start_inner
mgr._start_inner = fake_inner
try:
    ok2 = mgr.start("OuterModel", _real_load())
finally:
    mgr._start_inner = orig_inner
assert ok2 is True, ok2
assert ("inner-leaf-reached", "InnerModel") in calls, f"重入被误拦：{calls}"
assert mgr._loading_tid is None, "外层 finally 应已解锁"
# 不查 last_error：真实流程里 _start_inner 一进锁就会 self._error="" 清掉
# 拒绝时留下的文案，而桩函数绕过了那段；重入放行的证据看 calls 就够了。
print(f"2  OK 同线程重入放行：{calls}")

# ---------- 3. 异常路径也解锁 ----------
def boom(*a, **k):
    raise RuntimeError("模拟加载中途爆炸")
orig_inner = mgr._start_inner
mgr._start_inner = boom
mgr._loading_tid = None
try:
    mgr.start("X", _real_load())
    raise AssertionError("异常应向上抛")
except RuntimeError:
    pass
finally:
    mgr._start_inner = orig_inner
assert mgr._loading_tid is None and not mgr.busy_loading, "异常后标志必须解除"
print("3  OK 异常路径 finally 解锁，busy_loading=False")

# ---------- 4. 空闲时正常走 503（与 409 区分） ----------
assert not mgr.busy_loading
code, body = post_use("NoSuchModel_XYZ")
assert code == 503 and not body.get("busy"), (code, body)
print(f"4  OK 无人在加载时是真失败路径：503 error={body['error']!r}")

# ---------- 5. 前端守卫代码在页面里 ----------
html = urllib.request.urlopen(f"{BASE}/", timeout=5).read().decode("utf-8")
assert "if(switching)" in html and 'switching=""' in html, "前端守卫没进页面"
print("5  OK 前端 useModel 守卫已随页面下发")

ui.stop()
print("LOAD GUARD SMOKE OK")
