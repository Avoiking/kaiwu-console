# -*- coding: utf-8 -*-
"""MoE fit 扫档的**耗时构成**测量 + (档位 → 专家切分) 曲线采集。

为什么做这个（2026-10-21）：
  扫档是 kaiwu 最贵的一步（每档 = 真加载 + 7 枪测速），用户要干等十几分钟。
  想知道两件事：
    ① 真实耗时构成 —— 一档到底多久？加载占多少、测速占多少？实际扫了几档？
    ② 档位 → 专家切分（n_part）的曲线**能不能预先算出来**？
       如果能，就能"先算再小范围测"，把扫档从全量变成局部。

用法（**会占 GPU**，实测约 25~35 分钟）：
    python tests/_test_fit_timing.py                 # 全套
    python tests/_test_fit_timing.py --quick         # 只跑 3 档（快速看曲线形状）
    python tests/_test_fit_timing.py --ctx 32768     # 换窗口重跑

它**不写** config.json（只 start，不调 measure_ctx），跑完自动清理进程。
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "_scratch_quarantine"))
os.chdir(str(REPO))

from app.config import load_config, resolve_llama_server  # noqa: E402
from app.kaiwu_manager import KaiwuManager, _FIT_CANDIDATES  # noqa: E402

MODEL = "gemma-4-26B-A4B-it-UD"
OUT = REPO / "tests" / f"_fit_timing_{time.strftime('%Y%m%d_%H%M%S')}.json"

# 上游 fit 的 trace 日志里，它内部试过的**每个候选**都会打一行：
#   id=0, n_layer=31, n_part=26, overflow_type=4, mem= 6394 MiB
# 这一串就是「档位 → 专家切分 → 占多少显存」的阶梯（模型固有属性）。
_RE_LADDER = re.compile(
    r"n_layer=\s*(\d+),\s*n_part=\s*(\d+),\s*overflow_type=(\d+),\s*mem=\s*(\d+)\s*MiB")
_RE_FREE = re.compile(r"vs\.\s*(\d+)\s*MiB of free device memory")
_RE_TARGET = re.compile(r"id=0,\s*target=(\d+)\s*MiB")


def parse_fit_log(log: str) -> dict:
    """从上游 fit 的日志里抽出：空闲显存、该档预算、mem(n_part) 阶梯。

    ⚠️ 必须按 n_layer 过滤：日志里有一行 `n_layer=0, n_part=0`（"把所有 MoE
    tensor 搬到系统内存"的探测），它不是层数配置，混进来会让阶梯失真。
    只取 n_layer 等于**总层数**的那组候选。
    """
    raw: list[tuple[int, int, int]] = []
    for n_layer, n_part, mem in _RE_LADDER.findall(log):
        raw.append((int(n_layer), int(n_part), int(mem)))
    ladder: dict[int, int] = {}
    if raw:
        top = max(r[0] for r in raw)          # 模型总层数
        for n_layer, n_part, mem in raw:
            if n_layer != top:
                continue
            prev = ladder.get(n_part)
            # 同一 n_part 可能有多行（不同 overflow_type），取 mem 最小的那条
            if prev is None or mem < prev:
                ladder[n_part] = mem
    m_free = _RE_FREE.search(log)
    m_tgt = _RE_TARGET.search(log)
    return {
        "free_mib": int(m_free.group(1)) if m_free else None,
        "target_mib": int(m_tgt.group(1)) if m_tgt else None,
        "ladder": dict(sorted(ladder.items())),
    }


def _mem_at(ladder: dict[int, int], n: float) -> float:
    """阶梯上的分段线性取值（两端线性外推）。mem 随 n_part 单调递减。"""
    ns = sorted(ladder)
    if len(ns) == 1:
        return float(ladder[ns[0]])
    if n <= ns[0]:
        a, b = ns[0], ns[1]
    elif n >= ns[-1]:
        a, b = ns[-2], ns[-1]
    else:
        a = b = None
        for i in range(len(ns) - 1):
            if ns[i] <= n <= ns[i + 1]:
                a, b = ns[i], ns[i + 1]
                break
        if a is None:
            return float(ladder[ns[-1]])
    slope = (ladder[b] - ladder[a]) / (b - a)
    return ladder[a] + slope * (n - a)


def predict_n_part(ladder: dict[int, int], free_mib: int, fitt: int,
                    n_layer: int) -> tuple[int | None, str]:
    """给定「mem(n_part) 阶梯」和目标预算，算出预测 n_part。

    物理含义（2026-10-21 在 gemma 与 LFM2.5 上逐档验证）：
      · n_part = 有多少层的专家权重留在 CPU；**越小 = 专家越多在显存 = 越快**。
      · 上游 fit 的预算 = 运行时空闲显存 − fit_target（预留）。
      · 上游把层逐个从「只有稠密权重在显存」升级成「整层专家也在显存」，
        每升一层 n_part 减 1，**直到装不下为止**。
      ⇒ 所以答案就是：**让 mem(n_part) ≤ 预算 的最小 n_part**。

    返回 (预测 n_part, 说明)。返回 None 表示「连最省的配置都装不下」
    → 走层级别回填路径（层数不满，必然更慢）。
    """
    if not ladder or not free_mib:
        return None, "没有阶梯或空闲显存"
    target = free_mib - fitt
    hi = max(max(ladder), int(n_layer or 0))
    for n in range(0, hi + 1):
        if _mem_at(ladder, n) <= target:
            return n, (f"预算 {target} MiB（空闲 {free_mib} − 预留 {fitt}）；"
                       f"mem(n_part={n})={_mem_at(ladder, n):.0f} ≤ 预算")
    return None, (f"预算 {target} MiB 连最省配置（n_part={hi} 需 "
                  f"{ladder[max(ladder)]} MiB）都装不下 → 层级别回填")


def say(s: str = "") -> None:
    print(s, flush=True)


def main() -> int:
    quick = "--quick" in sys.argv
    global MODEL
    ctx = 114688
    for i, a in enumerate(sys.argv):
        if a == "--ctx" and i + 1 < len(sys.argv):
            ctx = int(sys.argv[i + 1])
        if a == "--model" and i + 1 < len(sys.argv):
            MODEL = sys.argv[i + 1]

    ladder = list(_FIT_CANDIDATES)
    if quick:
        ladder = [4096, 1024, 256]

    cfg = load_config()
    mgr = KaiwuManager()
    say(f"[环境] llama-server = {resolve_llama_server(cfg)}")
    say(f"[环境] 总显存 = {mgr.total_vram_mib()} MiB")
    say(f"[环境] 目标窗口 = {ctx}")
    say(f"[环境] 档位梯子 = {ladder}")

    rows: list[dict] = []

    def run_one(fitt: int, c: int, tag: str = "") -> dict:
        """加载一档 → 真测速 → 记录账目与耗时。返回一条记录。"""
        t_load0 = time.time()
        ok = mgr.start(MODEL, load_config(), timeout=600, ctx_override=c,
                       extra_args=["-fitt", str(fitt)])
        t_load = time.time() - t_load0
        if not ok:
            say(f"  [-fitt {fitt} @ {c}] 起不来（{mgr.last_error[:80]!r}）")
            mgr.stop(load_config())
            return {"fitt": fitt, "ctx": c, "ok": False, "load_s": round(t_load, 1),
                    "measure_s": 0.0, "tag": tag}
        acct = mgr._parse_vram_log() or {}
        off = mgr._gpu_offload_info() or (None, None)
        # 上游 fit 在这次加载里试过的**每个候选**（n_part → 占多少显存）——
        # 这是「这个模型自己的」切分阶梯，每个模型/每次运行都现测。
        fitlog = parse_fit_log(mgr.log_text())
        t_m = time.time()
        tps = mgr._measure_tps(load_config(), rounds=7, drop=2, stat="mean")
        t_meas = time.time() - t_m
        samples = list(mgr._last_tps_samples or [])
        used = sum(float(acct.get(k) or 0)
                   for k in ("model_mib", "kv_mib", "compute_mib", "mmproj_mib"))
        row = {
            "fitt": fitt, "ctx": c, "ok": True, "tag": tag,
            "load_s": round(t_load, 1), "measure_s": round(t_meas, 1),
            "total_s": round(t_load + t_meas, 1),
            "tps": round(tps, 2) if tps else None,
            "samples": [round(x, 2) for x in samples],
            "n_part": int(acct.get("n_part") or 0) or None,
            "on_gpu": off[0], "total_layers": off[1],
            "vram_mib": round(used, 1),
            "vram_pct": round(mgr._vram_pct(acct), 1),
            "model_mib": acct.get("model_mib"),
            "kv_mib": acct.get("kv_mib"),
            "compute_mib": acct.get("compute_mib"),
            "cpu_mapped_mib": acct.get("cpu_mapped_mib"),
            "fit_free_mib": fitlog.get("free_mib"),
            "fit_target_mib": fitlog.get("target_mib"),
            "fit_ladder": fitlog.get("ladder") or {},
        }
        say(f"  [-fitt {fitt} @ {c}]{tag} 加载 {row['load_s']}s + 测速 {row['measure_s']}s"
            f" = {row['total_s']}s | n_part={row['n_part']}"
            f" 层 {row['on_gpu']}/{row['total_layers']}"
            f" | 显存 {row['vram_mib']}MiB ({row['vram_pct']}%)"
            f" | {row['tps']} tok/s | 采样 {row['samples']}")
        if fitlog.get("ladder"):
            say(f"      上游fit: 空闲 {fitlog['free_mib']}MiB 预算 {fitlog['target_mib']}MiB"
                f" 阶梯 {fitlog['ladder']}")
        mgr.stop(load_config())
        time.sleep(1.0)
        return row

    try:
        # ---- 预热：把专家权重的页缓存喂热，否则"后跑的档天然更快"（顺序偏差）
        say("\n[预热] 加载一次 + 2 枪（喂热页缓存）…")
        t0 = time.time()
        ok = mgr.start(MODEL, load_config(), timeout=600, ctx_override=ctx,
                       extra_args=["-fitt", "1024"])
        say(f"[预热] 加载 {time.time()-t0:.0f}s ok={ok}")
        if ok:
            mgr._measure_tps(load_config(), rounds=2, drop=0)
            say(f"[预热] 预热采样 = {mgr._last_tps_samples}")
        mgr.stop(load_config())
        time.sleep(1.0)

        # ---- 主梯子
        say("\n[主梯子] 从保守到激进逐档真加载真测速")
        t_all = time.time()
        for fitt in ladder:
            rows.append(run_one(fitt, ctx))
        ladder_s = time.time() - t_all
        say(f"\n[主梯子] {len(ladder)} 档共 {ladder_s:.0f}s"
            f"（平均 {ladder_s/max(len(ladder),1):.0f}s/档）")

        # ---- 顺序偏差对照：把中间那档放到最后再测一次
        if not quick:
            say("\n[顺序偏差] 末尾重测 -fitt 512（和梯子里的读数比，差多少就是偏差）")
            rows.append(run_one(512, ctx, tag="repeat"))

        # ---- 不同窗口的加载耗时对照（KV 变大 → 加载更慢？）
        if not quick:
            say("\n[窗口对照] 同一档在 32768 窗口下的耗时")
            rows.append(run_one(512, 32768, tag="ctx32k"))
    finally:
        try:
            mgr.stop(load_config())
        except Exception:
            pass

    # ---------------- 汇总 ----------------
    say("\n" + "=" * 96)
    say(f"{'档位':>6} {'窗口':>8} {'加载s':>7} {'测速s':>7} {'合计s':>7} "
        f"{'n_part':>7} {'层':>8} {'显存MiB':>9} {'水位%':>7} {'tok/s':>8}")
    say("-" * 96)
    for r in rows:
        if not r.get("ok"):
            say(f"{r['fitt']:>6} {r['ctx']:>8} {r['load_s']:>7} {'起不来':>7}"
                f" {'':>7} {'':>7} {'':>8} {'':>9} {'':>7} {'':>8}")
            continue
        say(f"{r['fitt']:>6} {r['ctx']:>8} {r['load_s']:>7} {r['measure_s']:>7}"
            f" {r['total_s']:>7} {str(r['n_part']):>7}"
            f" {str(r['on_gpu'])+'/'+str(r['total_layers']):>8}"
            f" {r['vram_mib']:>9} {r['vram_pct']:>7} {str(r['tps']):>8}")

    # ---------------- 预测性分析：n_part 能不能预先算出来 ----------------
    say("\n" + "=" * 96)
    say(f"【预测性分析】{MODEL} —— 档位 → 专家切分（n_part）能否预先算出")
    say("=" * 96)
    main_rows = [r for r in rows if r.get("ok") and r.get("tag") != "repeat"
                  and r.get("ctx") == ctx]
    # 阶梯取任意一次加载的（它是模型固有属性，各次加载应当一致）
    ladder: dict[int, int] = {}
    free_mib = None
    for r in main_rows:
        if r.get("fit_ladder") and not ladder:
            ladder = dict(r["fit_ladder"])
        if r.get("fit_free_mib") and free_mib is None:
            free_mib = int(r["fit_free_mib"])
    if not ladder or not free_mib:
        say("❌ 没从日志里读到切分阶梯 / 空闲显存 → 无法预测")
    else:
        say(f"切分阶梯（本模型实测，mem 随 n_part 递减）：")
        say("  " + "  ".join(f"n_part={n}:{mem}MiB" for n, mem in ladder.items()))
        say(f"运行时空闲显存 = {free_mib} MiB")
        say("")
        say(f"{'档位':>6} {'预算MiB':>9} {'预测n_part':>10} {'实测n_part':>10} "
            f"{'判定':>8}  说明")
        ok_cnt = 0
        check_cnt = 0
        for r in main_rows:
            fitt = r["fitt"]
            pred, why = predict_n_part(ladder, free_mib, fitt,
                                       int(r.get("total_layers") or 0))
            actual = r.get("n_part")
            if actual is None:
                # 该档走的是「层级别回填」路径（连稠密层都放不下）
                judge = "跳过" if pred is None else "❗不符"
                say(f"{fitt:>6} {free_mib-fitt:>9} {str(pred):>10} {'层级回填':>10} "
                    f"{judge:>8}  {why}")
                continue
            check_cnt += 1
            hit = (pred is not None and abs(pred - actual) <= 0)
            ok_cnt += 1 if hit else 0
            say(f"{fitt:>6} {free_mib-fitt:>9} {str(pred):>10} {actual:>10} "
                f"{'✓' if hit else '✗':>8}  {why}")
        say(f"\n⇒ 预测准确率：{ok_cnt}/{check_cnt}")
        if ok_cnt == check_cnt and check_cnt >= 3:
            say("⇒ ✅ 这个模型上「先算后测」成立：可据此剪枝，跳过 n_part 相同的档")
        else:
            say("⇒ ❌ 预测不够准 → 剪枝不能上线（保持全扫）")

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    say(f"\n原始数据：{OUT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        # 铁律：测完必须清干净，不许留给用户收尾
        try:
            from _cleanup import cleanup, env_report
            cleanup()
            env_report()
        except Exception as e:  # 清理件缺失也要把进程杀了
            print(f"[cleanup] 清理件异常：{e}", flush=True)
            os.system('taskkill /F /IM llama-server.exe /T >nul 2>&1')
            os.system('taskkill /F /IM ollama.exe /T >nul 2>&1')
