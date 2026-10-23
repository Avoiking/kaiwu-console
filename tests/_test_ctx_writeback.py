"""ctx 覆盖值读写回归：过期快照不得整包回写抹掉测量结果（2026-10-16）。

背景：Ornith-1.5-9B@Q4_K_M 重测出 45056 但 config.json 仍是 32768 的疑案
（Memorix obs:33）。webui 的 set/clear_model_ctx 原来拿调用方那份 cfg
整包 save_config —— 若快照早于 _save_measured 落盘，测量结果就被静默
回滚。同款病灶踩过一次（_remember_last 抹掉
过 ctx_measured，修法=以磁盘现状为基准）。

本测试锁住修法：写盘必须以磁盘现状为基准；调用方传什么旧快照，
都不该影响磁盘上别人的键。旧实现下 [1][2] 的幸存断言会红。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import webui

PASS = FAIL = 0


def ok(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}")


# ---- 内存假磁盘：假装 measure 刚把 45056 写好 ----
STORE = {
    "model_ctx_overrides": {"Ornith-1.5-9B": 65536},
    "ctx_measured": {"Ornith-1.5-9B@Q4_K_M": 45056},
    "ctx_measured_env": {"Ornith-1.5-9B@Q4_K_M": {"gpu": "Test GPU"}},
    "last_model": "Ornith-1.5-9B",
}
webui.load_config = lambda: dict(STORE)


def _save(c):
    STORE.clear()
    STORE.update(c)


webui.save_config = _save

ui = webui.WebUI(object())   # 这两个方法不碰 manager，桩即可

# 调用方拿着一份**过期快照**：测量前的旧值、更老的 last_model、缺 env 键
STALE = dict(STORE)
STALE["ctx_measured"] = {"Ornith-1.5-9B@Q4_K_M": 32768}
STALE["last_model"] = "SomeoneElse"
del STALE["ctx_measured_env"]

print("\n[1] clear_model_ctx：过期快照不得回滚测量结果")
out = ui.clear_model_ctx("Ornith-1.5-9B", STALE)
ok(STORE["ctx_measured"].get("Ornith-1.5-9B@Q4_K_M") == 45056,
   f"ctx_measured 幸存（实际 {STORE['ctx_measured'].get('Ornith-1.5-9B@Q4_K_M')}）")
ok("Ornith-1.5-9B" not in STORE["model_ctx_overrides"], "覆盖值已清")
ok(STORE.get("last_model") == "Ornith-1.5-9B", "快照里更老的 last_model 没被写回")
ok(STORE.get("ctx_measured_env", {}).get("Ornith-1.5-9B@Q4_K_M") == {"gpu": "Test GPU"},
   "快照里没有的键没被删掉")
ok("Ornith-1.5-9B" not in (out.get("model_ctx_overrides") or {}), "返回值反映清理")

print("\n[2] set_model_ctx：过期快照同样不得回滚")
out = ui.set_model_ctx("Qwen3.5-9B-UD", 49152, STALE)
ok(STORE["ctx_measured"].get("Ornith-1.5-9B@Q4_K_M") == 45056,
   f"ctx_measured 幸存（实际 {STORE['ctx_measured'].get('Ornith-1.5-9B@Q4_K_M')}）")
ok(STORE["model_ctx_overrides"].get("Qwen3.5-9B-UD") == 49152, "新手填值已写入")
ok("Ornith-1.5-9B" not in STORE["model_ctx_overrides"], "上一个 case 清掉的没复活")
ok(STORE.get("last_model") == "Ornith-1.5-9B", "last_model 仍是磁盘值")

print("\n[3] 返回值可直接喂给 start()（_use 的用法）")
ok((out.get("model_ctx_overrides") or {}).get("Qwen3.5-9B-UD") == 49152,
   "返回值带新手填值")
ok((out.get("ctx_measured") or {}).get("Ornith-1.5-9B@Q4_K_M") == 45056,
   "返回值带磁盘上的测量值")

print(f"\n===== {PASS} 通过 / {FAIL} 失败 =====")
sys.exit(1 if FAIL else 0)
