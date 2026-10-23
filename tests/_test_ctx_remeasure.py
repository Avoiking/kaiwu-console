"""「重新测量」后端链路测试（不开真实推理服务，全 mock）。

覆盖 _use 端点的 remeasure 分支：
  · remeasure=true → 清掉该模型的 model_ctx_overrides，再 start(force_measure=True)
  · remeasure=false + ctx 数字 → 写覆盖值，普通 start
  · remeasure 与 ctx 同时给 → remeasure 优先（ctx 被忽略）
"""
import threading
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import webui
from app.kaiwu_manager import KaiwuManager

PASS = FAIL = 0


def ok(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {label}")
    else:
        FAIL += 1
        print(f"  ✗ {label}")


class FakeManager:
    """只记录 start 被怎么调用，不真的起 llama-server。"""

    def __init__(self):
        self.calls = []
        self.cur = 49152

    def start(self, model, cfg=None, timeout=None, ctx_override=None,
              force_measure=False):
        self.calls.append({"model": model, "force_measure": force_measure,
                           "ctx_override": ctx_override})
        return True

    def cur_ctx(self):
        return self.cur

    last_error = ""


class FakeHandler:
    """只借 _use 的逻辑：把 webui._Handler._use 的函数体拿来跑。"""

    def __init__(self, cfg_in):
        self.ui = type("UI", (), {})()
        self.ui.manager = FakeManager()
        self.ui.clear_model_ctx = lambda m, c=None: webui.WebUI.clear_model_ctx(
            self.ui, m, c)
        self.ui.set_model_ctx = lambda m, c, cfg=None: webui.WebUI.set_model_ctx(
            self.ui, m, c, cfg)
        self._cfg = cfg_in
        self.status = None
        self.payload = None

    def _read_json(self):
        return self.payload

    def _json(self, status, obj):
        self.status, self.body = status, obj


def run_use(payload, cfg_in):
    store = dict(cfg_in)
    webui.load_config = lambda: dict(store)
    def _save(c):
        store.clear()
        store.update(c)
    webui.save_config = _save

    h = FakeHandler(store)
    h.payload = payload
    webui._Handler._use(h)
    return h, store


BASE = {
    "context_size": "",
    "ctx_auto": True,
    "model_ctx_overrides": {"Ornith-1.5-9B": 65536, "Qwen3-4B": 8192},
    "ctx_measured": {"Ornith-1.5-9B": 32768},
}

print("\n[1] remeasure=true → 清覆盖 + 强制测量")
h, store = run_use({"name": "Ornith-1.5-9B", "remeasure": True}, BASE)
ok(h.status == 200, f"返回 200（实际 {h.status}）")
ok(h.ui.manager.calls[0]["force_measure"] is True, "start 收到 force_measure=True")
ok("Ornith-1.5-9B" not in (store.get("model_ctx_overrides") or {}),
   "该模型的手填覆盖被清除")
ok(store["model_ctx_overrides"].get("Qwen3-4B") == 8192, "其他模型的覆盖不受影响")
ok(store["ctx_measured"].get("Ornith-1.5-9B") == 32768, "测量记录保留（失败可兜底）")
ok(h.body.get("remeasured") is True, "响应带 remeasured=true")

print("\n[2] remeasure=false + ctx → 写覆盖，普通 start")
h, store = run_use({"name": "Ornith-1.5-9B", "ctx": 49152}, BASE)
ok(h.ui.manager.calls[0]["force_measure"] is False, "force_measure=False")
ok(store["model_ctx_overrides"].get("Ornith-1.5-9B") == 49152, "手填值已写入")
ok(h.body.get("remeasured") is False, "响应 remeasured=false")

print("\n[3] remeasure 与 ctx 同时给 → remeasure 优先")
h, store = run_use({"name": "Ornith-1.5-9B", "ctx": 49152, "remeasure": True}, BASE)
ok(h.ui.manager.calls[0]["force_measure"] is True, "仍走强制测量")
ok("Ornith-1.5-9B" not in (store.get("model_ctx_overrides") or {}),
   "ctx 被忽略，覆盖被清除（不会白跑一次测量）")

print("\n[4] 非法 ctx 仍被拦截")
h, store = run_use({"name": "Ornith-1.5-9B", "ctx": 10}, BASE)
ok(h.status == 400, f"返回 400（实际 {h.status}）")
ok("上下文要在" in h.body.get("error", ""), "错误提示正确")

print("\n[5] 缺少 name 仍被拦截")
h, store = run_use({"remeasure": True}, BASE)
ok(h.status == 400 and "name" in h.body.get("error", ""), "提示缺少 name")

print("\n[6] 路径形式的模型名也能清掉覆盖")
cfg6 = dict(BASE)
cfg6["model_ctx_overrides"] = {"D:/models/Qwen3-4B.gguf": 8192, "keep": 4096}
h, store = run_use({"name": "D:/models/Qwen3-4B.gguf", "remeasure": True}, cfg6)
ok("D:/models/Qwen3-4B.gguf" not in (store.get("model_ctx_overrides") or {}),
   "按完整路径存的覆盖被清除")
ok(store["model_ctx_overrides"].get("keep") == 4096, "无关项保留")

print(f"\n===== {PASS} 通过 / {FAIL} 失败 =====")
sys.exit(1 if FAIL else 0)