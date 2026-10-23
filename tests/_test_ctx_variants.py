# -*- coding: utf-8 -*-
"""验证「上下文按量化档记录」（2026-10-15）。

场景：
  A. 旧配置迁移 —— 平名记录只在单量化档时采信（Ornith 真实目录）
  B. 多档共存 —— 旧平名记录拒信，切档没测过就安排自动测量
  C. 每档各记 —— 同模型不同量化档互不沿用；指纹过期重测
  D. UI 展示 —— _ctx_hint / ctx_info 按当前档出数
  E. 老用法兜底 —— 调用方传完整路径时按文件名查表的行为不变
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.kaiwu_manager import KaiwuManager, Path_stem   # noqa: E402
import app.webui as webui                                # noqa: E402

FAILED = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  got={got!r} want={want!r}"))
    if not ok:
        FAILED.append(name)


# 不启动看门狗线程的裸实例（只调静态/纯实例方法）
M = KaiwuManager.__new__(KaiwuManager)

# ---------- 真实登记状态（用户取消登记前的备份） ----------
# 这份备份只在本机开发时存在，别的机器 / CI 上没有 → 缺了就跳过依赖它的
# A、D3、F 三段（其余段不依赖它，照跑）。
ROOT = Path(__file__).resolve().parent.parent
_bak_cands = sorted(ROOT.glob("data/config.json.bak_*before_unregister*"))
BAK = _bak_cands[0] if _bak_cands else None
bak = json.loads(BAK.read_text(encoding="utf-8")) if BAK else None
HAVE_BAK = bak is not None
if not HAVE_BAK:
    print("INFO  没有本机备份 config.json.bak_*before_unregister*，"
          "跳过 A / D3 / F 三段（需要真实登记状态）")

print("== A. 旧配置迁移（单量化档 → 平名记录无歧义，采信） ==")
if HAVE_BAK:
    check("A1 key(Ornith) = 模型@量化档",
          KaiwuManager._ctx_cache_key(bak, "Ornith-1.5-9B"), "Ornith-1.5-9B@Q4_K_M")
    check("A2 read(Ornith) 采信旧平名 32768",
          KaiwuManager._ctx_read(bak, "Ornith-1.5-9B"), (32768, "Ornith-1.5-9B"))
    check("A3 Ornith 是单量化档", KaiwuManager._ctx_single_variant(bak, "Ornith-1.5-9B"), True)
    check("A4 read(gemma) 从没测过 → 0",
          KaiwuManager._ctx_read(bak, "gemma-4-26B-A4B-it-UD"),
          (0, "gemma-4-26B-A4B-it-UD@IQ3_S"))
    check("A5 _should_measure(gemma) 要测（无缓存+ctx_auto）",
          M._should_measure(bak, "gemma-4-26B-A4B-it-UD"), True)
    # 软检查：本机指纹是否还对得上（对不上 = 会自动重测，也是正确行为）
    print(f"INFO  A6 _ctx_cache_valid(Ornith 旧指纹) = "
          f"{M._ctx_cache_valid(bak, 'Ornith-1.5-9B')}")

# ---------- 合成多量化档模型 ----------
tmp = Path(tempfile.mkdtemp(prefix="ctxvar_"))
d = tmp / "TestModel"
d.mkdir()
q4, q8 = d / "TestModel-Q4_K_M.gguf", d / "TestModel-Q8_0.gguf"
q4.write_bytes(b"x")
q8.write_bytes(b"x")

cfg = {
    "imported_models": {"TestModel": str(q4)},
    "ctx_measured": {"TestModel": 65536},      # 旧平名记录（模拟老配置）
    "ctx_measured_env": {"TestModel": {}},
    "ctx_auto": True,
    "cache_type_k": "q8_0", "cache_type_v": "q8_0",
}

print("== B. 多档共存：旧平名记录拒信 ==")
check("B1 key 带档", KaiwuManager._ctx_cache_key(cfg, "TestModel"), "TestModel@Q4_K_M")
check("B2 read → 0（平名 65536 不可信）",
      KaiwuManager._ctx_read(cfg, "TestModel"), (0, "TestModel@Q4_K_M"))
check("B3 n_variants=2 同样拒信",
      KaiwuManager._ctx_read(cfg, "TestModel", n_variants=2), (0, "TestModel@Q4_K_M"))
check("B4 _should_measure → True（自动补测）", M._should_measure(cfg, "TestModel"), True)
check("B5 _target_ctx 落 32768（不是 65536）",
      KaiwuManager._target_ctx(cfg, "TestModel"), 32768)
check("B6 _ctx_single_variant → False", KaiwuManager._ctx_single_variant(cfg, "TestModel"), False)

print("== B2. 切换到没测过的档 → 也要测 ==")
cfg_sw = json.loads(json.dumps(cfg))
cfg_sw["imported_models"]["TestModel"] = str(q8)
check("B7 read(Q8) → 0", KaiwuManager._ctx_read(cfg_sw, "TestModel"), (0, "TestModel@Q8_0"))
check("B8 _should_measure(Q8) → True", M._should_measure(cfg_sw, "TestModel"), True)

print("== C. 每档各记，互不沿用；指纹过期重测 ==")
fp = M._vram_fingerprint(cfg)      # 本机真实指纹
cfg_rec = json.loads(json.dumps(cfg))
cfg_rec["ctx_measured"] = {"TestModel@Q4_K_M": 40960, "TestModel@Q8_0": 51200}
cfg_rec["ctx_measured_env"] = {"TestModel@Q4_K_M": fp, "TestModel@Q8_0": fp}
check("C1 read(Q4) 用自己的 40960",
      KaiwuManager._ctx_read(cfg_rec, "TestModel"), (40960, "TestModel@Q4_K_M"))
check("C2 _should_measure(Q4) → False（测过记住）",
      M._should_measure(cfg_rec, "TestModel"), False)
check("C3 _target_ctx(Q4) = 40960",
      KaiwuManager._target_ctx(cfg_rec, "TestModel"), 40960)
cfg_sw2 = json.loads(json.dumps(cfg_rec))
cfg_sw2["imported_models"]["TestModel"] = str(q8)
check("C4 read(Q8) 用自己的 51200（不沿用 Q4 的 40960）",
      KaiwuManager._ctx_read(cfg_sw2, "TestModel"), (51200, "TestModel@Q8_0"))
check("C5 _target_ctx(Q8) = 51200",
      KaiwuManager._target_ctx(cfg_sw2, "TestModel"), 51200)
# 指纹过期（KV 量化档改了）→ 重测
fp_old = dict(fp)
fp_old["kv"] = "f16/f16"
cfg_stale = json.loads(json.dumps(cfg_rec))
cfg_stale["ctx_measured_env"] = {"TestModel@Q4_K_M": fp_old, "TestModel@Q8_0": fp_old}
check("C6 指纹过期 → _should_measure → True",
      M._should_measure(cfg_stale, "TestModel"), True)

print("== D. UI 展示（_ctx_hint） ==")
h = webui._ctx_hint(cfg_rec, "TestModel", str(q4), 2)
check("D1 有记录：measured/变体标签/stale",
      (h["measured"], h["variant"], h["stale"]), (40960, "Q4_K_M", False))
h2 = webui._ctx_hint(cfg, "TestModel", str(q4), 2)
check("D2 新档没测过：measured=0 + auto 提示",
      (h2["measured"], h2["auto"]), (0, True))
if HAVE_BAK:
    h3 = webui._ctx_hint(bak, "Ornith-1.5-9B", bak["imported_models"]["Ornith-1.5-9B"], 1)
    check("D3 旧平名采信：measured=32768，variant 空（平名键）",
          (h3["measured"], h3["variant"]), (32768, ""))

print("== E. 老用法兜底：传完整路径当模型名 ==")
p = tmp / "foo.gguf"
p.write_bytes(b"x")
cfg_path = {"ctx_measured": {"foo": 40000}, "ctx_auto": True}
check("E1 路径参数按文件名查到旧记录",
      KaiwuManager._target_ctx(cfg_path, str(p)), 40000)
check("E2 平名模型名不会误触兜底（B5 已覆盖）",
      KaiwuManager._target_ctx(cfg, "TestModel"), 32768)

print("== F. 无量化后缀文件名 → 键就是模型名 ==")
if HAVE_BAK:
    josie = bak["imported_models"]["josie-4b-instruct-32k"]
    check("F1 key(josie) = 模型名本身",
          KaiwuManager._ctx_cache_key(bak, "josie-4b-instruct-32k"), "josie-4b-instruct-32k")
check("F2 _ctx_cache_key 显式传 path 不查 cfg",
      KaiwuManager._ctx_cache_key({}, "X", str(q8)), "X@Q8_0")

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项失败：{FAILED}")
    sys.exit(1)
print("✓ 全部通过")
