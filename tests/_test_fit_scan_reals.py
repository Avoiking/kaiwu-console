# -*- coding: utf-8 -*-
"""真机验收：「先算后测」新路径 vs 强制全量扫描（老路径）—— 必须选出同一个档。

用法（**占 GPU，一次约 25~35 分钟**，两个模型各跑两条路径）：
    python tests/_test_fit_scan_reals.py LFM2.5-24B-A2B-UD
    python tests/_test_fit_scan_reals.py gemma-4-26B-A4B-it-UD

要点：
  · 每跑一次 measure_ctx 都会写 config.json（ctx_measured / fit_target）→
    跑之前整份备份，跑完还原，**不留任何污染**；
  · 老路径用 monkeypatch 把 fit_scan.parse_fit_ladder 打瞎 → 计划为空 →
    自动退回全量扫描（生产代码里就是这条兜底路径，不是假的对照）；
  · 真机上下文里记录每条路径实测了哪几档、各花了多久、最终选哪档。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "_scratch_quarantine"))
os.chdir(str(REPO))

from app import fit_scan  # noqa: E402
from app.config import config_path, load_config  # noqa: E402
from app.kaiwu_manager import KaiwuManager  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "LFM2.5-24B-A2B-UD"


def say(s: str = "") -> None:
    print(s, flush=True)


def run(label: str, blind: bool) -> dict:
    """跑一遍 measure_ctx。blind=True → 抹掉切分阶梯 → 走全量扫描（老路径）。"""
    mgr = KaiwuManager()
    mgr._logbuf.clear()
    # ⚠️ _logbuf 是 deque(maxlen=2000)：一次测量 8+ 次加载 × 每次约 274 行
    #   = 2200+ 行，**扫档阶段的日志会被挤掉**（2026-10-21 踩过，看起来像
    #   「没走先算后测」）。所以这里旁路一份无上限的日志。
    tee: list[str] = []
    orig_log = mgr._log

    def _tee(line):
        tee.append(line)
        return orig_log(line)

    mgr._log = _tee
    saved = fit_scan.parse_fit_ladder
    if blind:
        fit_scan.parse_fit_ladder = lambda log: {
            "free_mib": None, "ladder": {}, "n_layer": None}
    saved_fit = None
    orig_save = mgr._save_measured

    def _save(cfg, model, ctx, note="", mode="", fit=None):
        nonlocal saved_fit
        saved_fit = fit
        return orig_save(cfg, model, ctx, note=note, mode=mode, fit=fit)

    mgr._save_measured = _save
    # 两条路径都把「窗口二分」关掉（ctx_bisect_max_loads=0）：那段是**本次改动
    # 没碰的代码**，两条路径完全相同，验收只关心「扫档」这一段。关掉它能省下
    # 每条路径 3 次加载（≈8 分钟）。
    cfg = dict(load_config())
    cfg["ctx_bisect_max_loads"] = 0
    t0 = time.time()
    try:
        res = mgr.measure_ctx(MODEL, cfg)
    finally:
        fit_scan.parse_fit_ladder = saved
    dt = time.time() - t0
    log = "\n".join(tee)
    # 最直接的证据：每次加载带了什么 -fitt
    scanned = [(a.get("requested"), (a.get("extra") or [None, None])[1],
                a.get("ok"), a.get("n_part"), a.get("tps"))
               for a in (res.get("attempts") or [])
               if a.get("extra")]
    plan_line = next((l for l in tee if "只测这" in l), "")
    reason_line = next((l for l in tee if "先算后测不可用" in l), "")
    used_plan = "先算后测" in log and "只测这" in log
    say(f"\n===== {label} =====")
    say(f"用时 {dt:.0f}s（{dt/60:.1f} 分钟）")
    say(f"加载总次数 {res.get('loads')}")
    say(f"带 -fitt 的加载：{scanned}")
    say(f"走的是：{'先算后测（' + plan_line.strip() + '）' if used_plan else '全量扫描'}"
        + (f"　原因：{reason_line.strip()}" if reason_line else ""))
    say(f"最终选中 fit_target = {saved_fit}")
    say(f"最终窗口 best = {res.get('best')}")
    mis = [l for l in tee if "预测失准" in l]
    say(f"预测失准：{'有（' + str(len(mis)) + ' 条）' if mis else '无'}")
    return {"label": label, "seconds": round(dt, 1), "scanned": scanned,
            "fit": saved_fit, "best": res.get("best"),
            "plan": used_plan, "loads": res.get("loads"),
            "reason": reason_line, "mismatch": mis,
            "log": log}


def main() -> int:
    cfgp = config_path()
    bak = cfgp.parent / f"config.json.bak_reals_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(cfgp, bak)
    say(f"[备份] {bak.name}")
    out = {}
    try:
        out["new"] = run("新路径：先算后测", blind=False)
        out["full"] = run("对照：全量扫描（老路径）", blind=True)
    finally:
        shutil.copy2(bak, cfgp)
        bak.unlink(missing_ok=True)
        say(f"\n[还原] config.json 已从备份还原（{bak.name} 已删除）")

    say("\n" + "=" * 72)
    n, f = out["new"], out["full"]
    say(f"{'':22}{'新路径':>18}{'全量扫描':>18}")
    say(f"{'加载总次数':22}{str(n['loads']):>18}{str(f['loads']):>18}")
    say(f"{'带 -fitt 的档位':22}"
        f"{str([x[1] for x in n['scanned']]):>18}{str([x[1] for x in f['scanned']]):>18}")
    say(f"{'选中的 fit_target':22}{str(n['fit']):>18}{str(f['fit']):>18}")
    say(f"{'最终窗口':22}{str(n['best']):>18}{str(f['best']):>18}")
    say(f"{'耗时(秒)':22}{n['seconds']:>18}{f['seconds']:>18}")
    same = n["fit"] == f["fit"]
    faster = n["seconds"] < f["seconds"]
    say("=" * 72)
    say(f"选出同一个档：{'✓ 是' if same else '✗ 否'}")
    say(f"新路径更快：{'✓ 是' if faster else '✗ 否'}"
        f"（省 {f['seconds']-n['seconds']:.0f}s，"
        f"{100*(f['seconds']-n['seconds'])/max(f['seconds'],1):.0f}%）")
    say(f"新路径预测失准：{'有' if n['mismatch'] else '无'}")
    say(f"全量扫描预测失准：{'有' if f['mismatch'] else '无'}")
    return 0 if same else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        try:
            from _cleanup import cleanup, env_report
            cleanup()
            env_report()
        except Exception as e:
            print(f"[cleanup] 清理件异常：{e}", flush=True)
            os.system('taskkill /F /IM llama-server.exe /T >nul 2>&1')
            os.system('taskkill /F /IM ollama.exe /T >nul 2>&1')
