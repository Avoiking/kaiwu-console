# -*- coding: utf-8 -*-
"""「先算后测」扫档（app/fit_scan.py）的规格测试 —— 不开推理服务。

锁住的行为（2026-10-21，方案见 memorix #109/#110）：
  1. 切分阶梯解析：只认总层数那一组候选（`n_layer=0` 的探针行必须滤掉，
     否则预测全退化成 0）；同一 n_part 取 mem 最小；单调性兜底；
  2. 只取**最近一次加载**的 fit 段（日志跨加载累积，空闲显存会变）；
  3. mem_at 分段线性（含两端外推）；predict_n_part = "装得下的最小 n_part"；
  4. 预测精度：LFM2.5 真机 7 档，用**单次加载**的阶梯 6/7、用**累积合并**的
     阶梯 7/7（差的那一个是保守极端的层级别边界，差 1 档且不参与选档）；
  5. plan_scan：预测最快的档 + 左右各一邻档；同 n_part 跳过；超水位跳过；
     没有阶梯 → 空计划（调用方退回全扫）；同时给出**每个**候选档的预测
     n_part（调用方要用它复核「跳过」的判断有没有错）；
  6. **端到端**：measure_ctx 走先算后测只测 2 档，且选出的档位与全量扫描一致；
     预测与实测差 >1 层 → **从头**全量扫描（不是补扫剩余档 —— 那会造成
     「先激进后保守」的混合顺序，页缓存热度偏差方向翻转）；
     计划档起不来 → 回退全扫；
  7. **复核**：计划档测完后阶梯变完整 → 用累积阶梯重算被跳过那几档的预测，
     预测变了就补测那档（当初「跳过」的理由可能是基于浅阶梯做的判断）。

真机数据来源（tests/_fit_timing_20261021_204451.json）：
  LFM2.5-24B-A2B-UD@Q3_K_XL @118784、41 层、空闲 6944 MiB
  4096: n_part 39 / 30.94 tok/s    2048: 32 / 37.93    1024: 28 / 40.87
  512: 26 / 42.51    256: 25 / 43.81    128: 24 / 45.52    64: 24 / 44.96
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.kaiwu_manager as _km
from app import fit_scan
from app.kaiwu_manager import KaiwuManager, _FIT_CANDIDATES

PASS = FAIL = 0


def ok(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}")


# ---- 真机日志夹具（2026-10-21 捕获，LFM2.5-24B-A2B-UD -fitt 512 @118784）----
# 关键点：n_layer=0 那行是"把所有 MoE tensor 搬到系统内存"的探测，must 滤掉；
# 各档 `id=0, target=` 的数不可信（各档打 5920/4896/6432…，与 free−fitt 不符）。
FIT_LOG = [
    "0.00.299.392 I cmn  common_init_: fitting params to device memory ...",
    "0.00.709.505 I common_params_fit_impl: projected to use 12438 MiB of device memory vs. 6944 MiB of free device memory",
    "0.00.709.513 I common_params_fit_impl: cannot meet free memory target of 512 MiB, need to reduce device memory by 6006 MiB",
    "0.01.135.404 I common_params_fit_impl: id=0, target=6432 MiB",
    "0.01.554.407 I common_params_fit_impl: id=0, n_layer= 0, n_part= 0, overflow_type=4, mem=   945 MiB",
    "0.02.014.549 I common_params_fit_impl: id=0, n_layer=41, n_part=40, overflow_type=4, mem=  2751 MiB",
    "0.02.440.125 I common_params_fit_impl: id=0, n_layer=41, n_part= 0, overflow_type=4, mem= 12438 MiB",
    "0.02.896.570 I common_params_fit_impl: id=0, n_layer=41, n_part=25, overflow_type=4, mem=  6462 MiB",
    "0.03.356.074 I common_params_fit_impl: id=0, n_layer=41, n_part=26, overflow_type=4, mem=  6213 MiB",
    "0.03.810.617 I common_params_fit_impl: id=0, n_layer=41, n_part=26, overflow_type=2, mem=  6286 MiB",
    "0.04.263.767 I common_params_fit_impl: id=0, n_layer=41, n_part=26, overflow_type=3, mem=  6360 MiB",
    "0.04.263.814 I common_fit_params: successfully fit params to free device memory",
    "0.04.263.822 I common_fit_params: fitting params to free memory took 3.96 seconds",
]
# 后来几档（目标更小 → 试探更靠激进端）把阶梯补全的点
FIT_LOG_RICH = [
    "0.00.299.392 I cmn  common_init_: fitting params to device memory ...",
    "0.00.709.505 I common_params_fit_impl: projected to use 12800 MiB of device memory vs. 6944 MiB of free device memory",
    "0.01.554.407 I common_params_fit_impl: id=0, n_layer= 0, n_part= 0, overflow_type=4, mem=   945 MiB",
    "0.02.014.549 I common_params_fit_impl: id=0, n_layer=41, n_part=38, overflow_type=4, mem=  3000 MiB",
    "0.02.060.000 I common_params_fit_impl: id=0, n_layer=41, n_part=39, overflow_type=4, mem=  2751 MiB",
    "0.02.100.000 I common_params_fit_impl: id=0, n_layer=41, n_part=27, overflow_type=4, mem=  5964 MiB",
    "0.02.200.000 I common_params_fit_impl: id=0, n_layer=41, n_part=28, overflow_type=4, mem=  5715 MiB",
    "0.02.300.000 I common_params_fit_impl: id=0, n_layer=41, n_part=24, overflow_type=4, mem=  6711 MiB",
    "0.02.400.000 I common_params_fit_impl: id=0, n_layer=41, n_part=23, overflow_type=4, mem=  6960 MiB",
    "0.04.263.822 I common_fit_params: fitting params to free memory took 3.96 seconds",
]

# 真机（LFM @118784）：档位 → n_part / tok/s
LFM_TRUTH = [(4096, 39, 30.94), (2048, 32, 37.93), (1024, 28, 40.87),
             (512, 26, 42.51), (256, 25, 43.81), (128, 24, 45.52),
             (64, 24, 44.96)]
FREE = 6944
TOTAL_VRAM = 8151
KaiwuManager.total_vram_mib = staticmethod(lambda: TOTAL_VRAM)


def log_of(lines):
    return "\n".join(lines)


print("\n[1] 切分阶梯解析（真机日志原文）")
p1 = fit_scan.parse_fit_ladder(log_of(FIT_LOG))
ok(p1["free_mib"] == 6944, f"空闲显存取运行时的 6944 MiB（实际 {p1['free_mib']}）")
ok(p1["n_layer"] == 41, f"总层数 41（实际 {p1['n_layer']}）")
ok(p1["ladder"].get(0) == 12438,
   f"n_layer=0 的探针行（mem=945）被滤掉，n_part=0 只留『专家全在显存』的 12438"
   f"（否则预测会退化成 0；实际 {p1['ladder'].get(0)}）")
ok(p1["ladder"].get(26) == 6213,
   f"同 n_part=26 的三行取 mem 最小 6213（实际 {p1['ladder'].get(26)}）")
ok(set(p1["ladder"]) == {0, 25, 26, 40},
   f"只留总层数那一组候选 {sorted(p1['ladder'])}")

print("\n[2] 只认最近一次加载的 fit 段（日志跨加载累积）")
old = FIT_LOG[:-2] + ["common_fit_params: fitting params to free memory took 3.96 seconds"]
p2 = fit_scan.parse_fit_ladder(log_of(old + FIT_LOG_RICH))
ok(p2["free_mib"] == 6944, f"空闲显存取最近一次（实际 {p2['free_mib']}）")
ok(24 in p2["ladder"] and 25 not in p2["ladder"],
   f"阶梯只来自最后一段（实际 {sorted(p2['ladder'])}）")
tail = log_of(["0.00.1 I common_init_: fitting params to device memory ...",
               "projected to use 9000 MiB of device memory vs. 5000 MiB of free device memory"])
ok(fit_scan.parse_fit_ladder(tail)["free_mib"] == 5000,
   "旧加载的 6944 不会盖掉新加载的 5000（它就在新段里）")
ok(fit_scan.parse_fit_ladder("") == {"free_mib": None, "ladder": {}, "n_layer": None},
   "空日志 → 空阶梯（调用方据此退回全扫）")

print("\n[3] 阶梯合并（后面的档给出更丰富的点）")
merged = fit_scan.merge_ladder(p1["ladder"], p2["ladder"])
ok(set(merged) == {0, 23, 24, 25, 26, 27, 28, 38, 39, 40},
   f"合并后点数变多（实际 {sorted(merged)}）")
ok(fit_scan.merge_ladder({26: 6213}, {26: 6300})[26] == 6213,
   "同 n_part 取 mem 最小（不被后一次的大值覆盖）")

print("\n[4] mem_at：分段线性 + 两端外推")
lad = merged
ok(fit_scan.mem_at(lad, 25) == 6462, "阶梯点原值不插值变形")
ok(abs(fit_scan.mem_at(lad, 25.5) - (6462 + 6213) / 2) < 1, "两点之间线性插值")
ok(abs(fit_scan.mem_at(lad, 27) - 5964) < 1, "落在已知点上")
ok(fit_scan.mem_at({10: 2000, 20: 1000}, 5) > 2000, "左端外推（n 小于最小点）")
ok(fit_scan.mem_at({10: 2000, 20: 1000}, 40) < 1000, "右端外推（n 大于最大点）")

print("\n[5] 预测 n_part = 「装得下的最小 n_part」")
ok(fit_scan.predict_n_part({}, FREE, 512)[0] is None, "没阶梯 → None（不猜）")
ok(fit_scan.predict_n_part(lad, None, 512)[0] is None, "没空闲显存 → None（不猜）")
n512, why512 = fit_scan.predict_n_part(lad, FREE, 512, 41)
ok(n512 == 26, f"fitt 512 → 预算 6432 → n_part 26（实际 {n512}）")
ok("6432" in why512, f"说明里给出预算算法（{why512}）")
# 单次加载的阶梯只有 {0,25,26,40}
hits_single = [fit_scan.predict_n_part(p1["ladder"], FREE, f, 41)[0] for f, _, _ in LFM_TRUTH]
truth = [t[1] for t in LFM_TRUTH]
ok(sum(1 for a, b in zip(hits_single, truth) if a == b) == 6,
   f"单次加载阶梯 → 6/7 命中（实际 {hits_single}）")
ok(hits_single[0] == 40 and truth[0] == 39,
   "唯一差的是最保守那档（层级别回填的边界，外推差 1 层；不参与选档）")
hits_merged = [fit_scan.predict_n_part(lad, FREE, f, 41)[0] for f, _, _ in LFM_TRUTH]
ok(hits_merged == truth,
   f"累积合并阶梯 → 7/7 零误差（实际 {hits_merged} vs 真机 {truth}）")
over, why_over = fit_scan.predict_n_part({20: 9000, 30: 8000}, 5000, 64, 41)
ok(over is None and "层级别回填" in why_over, f"连最省都装不下 → 层级别回填（{why_over}）")

print("\n[6] plan_scan：预测最快 + 左右各一邻档")
plan = fit_scan.plan_scan(lad, FREE, _FIT_CANDIDATES, n_layer=41,
                          total_vram_mib=TOTAL_VRAM)
fitts = [p["fitt"] for p in plan["plan"]]
ok(fitts == [256, 128], f"LFM 只需实测 256 / 128（实际 {fitts}）")
ok(plan["skipped"].get(64, "").startswith("与 128 同 n_part=24"),
   f"64 因与 128 同 n_part 被跳过（{plan['skipped'].get(64)}）")
ok(all(p["n_part"] == min(q["n_part"] for q in plan["plan"]) for p in plan["plan"][1:]),
   "选中的就是预测 n_part 最小的那档")
ok(plan["reason"] == "", "预测可用时 reason 为空")

print("\n[6b] gemma @32768 真机阶梯：邻档要归一到同 n_part 里更保守的那个")
# 2026-10-21 22:1x 真机：阶梯 {0:12442, 21:6566, 22:6263, 30:4129}，空闲 6946，31 层
# 预测 4096=None（层级别回填）/2048=28/1024=24/512=22/256=21/128=21/64=20
# ⚠️ 若把「64 的邻档」原样取成 128（同为 n_part=21 但水位高 2pp），就会拿更多显存
#    换一个噪声内的读数 —— 必须归一到 256。
GEMMA_LADDER = {0: 12442, 21: 6566, 22: 6263, 30: 4129}
gp = fit_scan.plan_scan(GEMMA_LADDER, 6946, _FIT_CANDIDATES, n_layer=31,
                        total_vram_mib=TOTAL_VRAM)
ok([p["fitt"] for p in gp["plan"]] == [256, 64],
   f"gemma 只测 256（n_part 21 里最保守的）和 64（预测最快）（实际 {[p['fitt'] for p in gp['plan']]}）")
ok(all(p["n_part"] == 21 for p in gp["plan"] if p["fitt"] == 256),
   "256 与 128 同 n_part=21 → 取 256")
ok(gp["skipped"].get(4096, "").startswith("预测为层级别回填"),
   f"4096 预测层级别回填 → 不测（{gp['skipped'].get(4096)}）")

print("\n[7] plan_scan 的三种退化")
ok(fit_scan.plan_scan({}, FREE, _FIT_CANDIDATES)["plan"] == [],
   "没阶梯 → 空计划（退回全扫）")
no_free = fit_scan.plan_scan(lad, None, _FIT_CANDIDATES)
ok(no_free["plan"] == [] and "空闲显存" in no_free["reason"],
   f"没空闲显存 → 空计划（{no_free['reason']}）")
# 全部同 n_part（小模型全装得下）→ 只测最保守那档（同配置分不出速度）
flat = {0: 2100, 41: 2000}
fp = fit_scan.plan_scan(flat, 7000, _FIT_CANDIDATES, n_layer=41,
                        total_vram_mib=TOTAL_VRAM)
ok([p["fitt"] for p in fp["plan"]] == [4096],
   f"所有档同 n_part → 只测最保守的 4096（实际 {[p['fitt'] for p in fp['plan']]}）")
# 激进档预测贴顶（>88% = 7173 MiB）→ 不进计划，选次激进的
tight = {0: 30000, 20: 7100, 21: 7300, 22: 7500}
tp = fit_scan.plan_scan(tight, 8000, _FIT_CANDIDATES, n_layer=41,
                        total_vram_mib=TOTAL_VRAM)
ok(all(p["pct"] is None or p["pct"] <= 88 for p in tp["plan"]),
   f"计划里没有预测超 88% 的档（{[(p['fitt'], p['pct']) for p in tp['plan']]}）")

# ============ [8] 端到端：measure_ctx 走先算后测 ============
print("\n[8] measure_ctx 端到端（桩驱动）：先算后测只测 2 档，且与全扫同结论")
_td = Path(tempfile.mkdtemp(prefix="fitscan_"))
(_td / "G-IQ3_S.gguf").write_bytes(b"x")
_orig_sleep = _km.time.sleep
_km.time.sleep = lambda *a, **k: None
CFG = {"ctx_vram_reserve_mib": 512, "ctx_estimate_ratio": 0.42,
       "ctx_moe_min_tps": 15.0, "ctx_dense_min_tps": 30.0,
       "cache_type_k": "q8_0", "cache_type_v": "q8_0",
       "ctx_tps_rounds": 7, "ctx_tps_tokens": 400, "ctx_tps_timeout_sec": 120}

NPART = {f: n for f, n, _ in LFM_TRUTH}
TPS = {f: t for f, _, t in LFM_TRUTH}
# 真机实测显存（MiB）—— 水位判据用（>88% = 7173 MiB）
VRAM_AT = {39: 2824.6, 32: 4866.6, 28: 5862.6, 26: 6360.6, 25: 6609.6, 24: 6784.6}
# 基线（不带 -fitt）那次加载打出来的阶梯：真机上就这几个点
BASE_LADDER = [(0, 12438), (25, 6462), (26, 6213), (40, 2751)]


def sim(inject_log=True, npart_fn=None, fail_on=(), hot_on=(), ladder_seq=None):
    """桩驱动跑一遍 measure_ctx。inject_log=True 时模拟 llama-server 打 fit 日志。

    ladder_seq：每次加载注入的阶梯（list of {n_part: mem}），按加载次序循环用。
    用来模拟「阶梯随目标变大而变丰富」→ 触发「复核被跳过的档」那条路径。
    """
    mgr = KaiwuManager()
    mgr._logbuf.clear()
    state = {"ctx": 0}
    starts, records = [], []
    cfg = dict(CFG, ctx_probe_ctx=32768, ctx_auto_min=32768, ctx_auto_max=131072,
               ctx_bisect_max_loads=3,
               imported_models={"G": str(_td / "G-IQ3_S.gguf")})
    mgr.stop = lambda cfg=None: True
    mgr._llama_supports = lambda cfg2, flag: flag in ("-fitt", "--cpu-moe")

    def _fitt_of(call):
        e = call.get("extra") or []
        return int(e[1]) if len(e) == 2 and e[0] == "-fitt" else None

    def _start(model, cfg2=None, timeout=None, ctx_override=None,
               force_measure=False, moe_cpu=False, extra_args=None):
        state["ctx"] = int(ctx_override or 0)
        starts.append({"ctx": state["ctx"], "moe_cpu": moe_cpu,
                       "extra": list(extra_args) if extra_args else None})
        fitt = _fitt_of(starts[-1])
        if fitt in fail_on:
            return False
        if inject_log:
            lad = (ladder_seq[min(len(starts) - 1, len(ladder_seq) - 1)]
                   if ladder_seq else BASE_LADDER)
            mgr._log("0.00.1 I cmn  common_init_: fitting params to device memory ...")
            mgr._log("0.00.2 I fit_impl: projected to use 12438 MiB of device memory "
                     f"vs. {FREE} MiB of free device memory")
            mgr._log("0.00.3 I fit_impl: id=0, target=6432 MiB")
            mgr._log("0.00.4 I fit_impl: id=0, n_layer= 0, n_part= 0, "
                     "overflow_type=4, mem=   945 MiB")
            for n, mem in lad:
                mgr._log(f"0.00.5 I fit_impl: id=0, n_layer=41, n_part={n:>2}, "
                         f"overflow_type=4, mem=  {mem} MiB")
            mgr._log("0.00.6 I common_fit_params: fitting params to free memory took 3 s")
        return True

    def _acct():
        fitt = _fitt_of(starts[-1])
        n = (npart_fn(fitt) if npart_fn else NPART.get(fitt, 31))
        return {"n_expert": 128, "n_expert_used": 8, "print_info_seen": True,
                "model_mib": 3693.0, "compute_mib": 896.0,
                "cpu_mapped_mib": 10425.2, "kv_mib": state["ctx"] * 10.6 / 1024.0,
                "kv_fixed_mib": 0.0, "per_token_kv_mib": 10.6 / 1024.0,
                "n_ctx": state["ctx"], "on_gpu": 41, "total_layers": 41,
                "n_part": n}

    def _pct(acct=None):
        fitt = _fitt_of(starts[-1])
        if fitt in hot_on:
            return 96.0
        n = (npart_fn(fitt) if npart_fn else NPART.get(fitt, 31))
        return VRAM_AT.get(n, 4000.0) / TOTAL_VRAM * 100

    mgr.start = _start
    mgr._gpu_offload_info = lambda: (41, 41)
    mgr._parse_vram_log = _acct
    mgr._vram_pct = _pct
    mgr._measure_tps = (lambda cfg2, tokens=None, timeout=None, rounds=None,
                        stat="max", drop=None:
                        TPS.get(_fitt_of(starts[-1]), 20.0))
    mgr._save_measured = (lambda cfg2, model, ctx, note="", mode="", fit=None:
                          records.append({"ctx": ctx, "fit": fit, "mode": mode}))
    mgr.granted_ctx = lambda cfg2=None: state["ctx"]
    res = mgr.measure_ctx("G", cfg)
    scan = [_fitt_of(c) for c in starts if c["ctx"] == 32768 and c["extra"]]
    return res, starts, records, list(mgr._logbuf), scan


try:
    # ---- 8a 先算后测：只测 256/128，选出 128（与全扫一致）----
    res_f, st_f, rec_f, lg_f, scans_f = sim()
    ok(scans_f == [256, 128], f"只实测预测的两档 256/128（实际扫了 {scans_f}）")
    ok(rec_f and rec_f[0]["fit"] == 128, f"选出 128（实际 {rec_f}）")
    ok(any("先算后测" in l for l in lg_f), "日志写明走的是先算后测")
    ok(any("跳过 -fitt 64" in l and "同 n_part" in l for l in lg_f),
       "日志写明 64 为什么没测（同 n_part，分不出速度差）")
    ok(any("只测这 2 档" in l for l in lg_f), "日志写明只测几档")
    ok(not any("预测失准" in l for l in lg_f), "预测与实测相符 → 不触发补扫")

    # ---- 8b 对照：日志里没有 fit 阶梯（老二进制）→ 全扫，结论必须一样 ----
    res_s, st_s, rec_s, lg_s, scans_s = sim(inject_log=False)
    ok(len(scans_s) > len(scans_f),
       f"没有阶梯时全扫（全扫 {scans_s} vs 先算后测 {scans_f}）")
    ok(rec_s[0]["fit"] == rec_f[0]["fit"] == 128,
       f"两条路径选出同一个档（全扫 {rec_s[0]['fit']} / 先算后测 {rec_f[0]['fit']}）")
    ok(any("先算后测不可用" in l for l in lg_s), "日志明说为什么退回全扫")

    # ---- 8c 预测失准 → 从头全量扫描（预测不参与选档）----
    bad = lambda f: (NPART.get(f, 31) - 3) if f in (256, 128) else NPART.get(f, 31)
    res_b, st_b, rec_b, lg_b, scans_b = sim(npart_fn=bad)
    ok(any("预测失准" in l for l in lg_b), "预测失准有日志（不静默省事）")
    # 差 >1 层才算失准（差 1 层是边界外推误差，不算）→ 从头按旧顺序重扫，
    # 不能变成「先激进后保守」的混合顺序（页缓存热度偏差方向会翻转）。
    ok(scans_b == [256, 4096, 2048, 1024, 512, 256, 128, 64],
       f"失准后**从头**按旧顺序全扫（实际扫了 {scans_b}）")
    ok(any("从头全量扫描" in l for l in lg_b), "日志明说从头全扫")

    # ---- 8d 计划档起不来 → 回退全量扫描 ----
    res_d, st_d, rec_d, lg_d, scans_d = sim(fail_on=(256,))
    ok(any("回退全量扫描" in l for l in lg_d),
       "计划档起不来 → 回退全扫（日志有回退说明）")
    ok(rec_d and rec_d[0]["fit"] == 128, f"回退后仍选出 128（实际 {rec_d}）")

    # ---- 8e 预测最快的档实测超水位 → 扫档就此收工（水位保护，行为不变）----
    res_h, st_h, rec_h, lg_h, scans_h = sim(hot_on=(128,))
    ok(any("水位保护" in l for l in lg_h), "超 88% 打出水位保护日志")
    ok(scans_h == [256, 128], f"超水位即停、不再往后测（实际 {scans_h}）")

    # ---- 8f 复核：被跳过的档，预测随阶梯变完整而改变 → 补测 ----
    # 场景：探测那次的阶梯很浅 → 64 与 128 预测同 n_part → 64 被跳过；
    # 计划档测完后阶梯多了个采样点（n_part=23, mem=6800）→ 64 的预测从 24
    # 变成 23（跟已测的 128 不再是同一配置）→ 当初的跳过可能是错判 → 补测。
    RICH = [(0, 12438), (23, 6800), (24, 6711), (25, 6462), (26, 6213), (40, 2751)]
    res_r, st_r, rec_r, lg_r, scans_r = sim(ladder_seq=[BASE_LADDER, RICH, RICH])
    ok(any("补测：-fitt" in l for l in lg_r),
       "阶梯变完整后预测变了 → 有补测动作（日志有说明）")
    ok(64 in scans_r, f"被跳过的 64 档被补测（实际扫了 {scans_r}）")
    # 反例：阶梯不变时不该有多余的补测
    res_n, st_n, rec_n, lg_n, scans_n2 = sim()
    ok(not any("补测：-fitt" in l for l in lg_n), "阶梯没变 → 不做多余补测")
    ok(scans_n2 == [256, 128], f"常规路径仍只测 2 档（实际 {scans_n2}）")
finally:
    _km.time.sleep = _orig_sleep

print(f"\n===== {PASS} 通过 / {FAIL} 失败 =====")
if FAIL:
    print("注：FAIL 说明「先算后测」的定位或兜底出了问题 —— 对照文件头的真机数据。")
sys.exit(1 if FAIL else 0)
