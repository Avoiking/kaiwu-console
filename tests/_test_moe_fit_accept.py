# -*- coding: utf-8 -*-
"""MoE 专家分配优化（-fitt 扫描）真机验收 —— 方案 §7 全项，可重复跑。

用法（**要占 GPU**，一次约 30~45 分钟；每档 7 枪是独立验收报告的要求）：
    python _test_moe_fit_accept.py            # 全部
    python _test_moe_fit_accept.py 7.1 7.1b   # 只跑指定小节

与临时脚本的区别：
  · 日志走 `_log` 打点（不读 `_logbuf` 环形缓冲 —— 它 maxlen=2000，
    多次加载后会淘汰早期行，2026-10-20 就因此丢了扫描档位日志）；
  · 每节给 PASS/FAIL 判定 + 末尾摘要，任一必修项 FAIL 则退出码 1；
  · 开头把 config.json 整体备份到 data/（同日只备一次），末尾打印差异。
只应在开发机上跑：它会真实写入 ctx_measured / fit_target / ctx_measured_note。
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(str(REPO))

from app.config import config_path, load_config, resolve_llama_server  # noqa: E402
from app.kaiwu_manager import KaiwuManager  # noqa: E402

GEMMA = "gemma-4-26B-A4B-it-UD"
ORNITH = "Ornith-1.5-9B"
OLD_EXE = str(Path.home() / ".kaiwu" / "bin" / "llama-server-cuda.exe")
# 老二进制（~/.kaiwu/bin 里那个）的启动检查**必须换一个它认识的架构**：~/.kaiwu/bin 那个构建（2026-05）
# 不认 `gemma4`（`unknown model architecture: 'gemma4'`，实测 2026-10-20 复现）——
# 拿 gemma 去验"老二进制不再被 --cors-origins 打死"永远会失败，且失败原因与旗标无关。
# 用 qwen35 的稠密模型（升级前就是它在跑老二进制）才测得到旗标这一层。
OLD_BIN_MODEL = "Qwen3.5-4B-UD"
LOGP = REPO / "_scratch_quarantine" / f"moe_fit_accept_{time.strftime('%Y%m%d_%H%M%S')}.log"

VERDICTS: list[tuple[str, bool, str]] = []


def say(s: str = "") -> None:
    print(s, flush=True)


def hdr(s: str) -> None:
    say("\n" + "=" * 74)
    say(s)
    say("=" * 74)


def verdict(name: str, passed: bool, detail: str = "") -> None:
    VERDICTS.append((name, bool(passed), detail))
    say(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))


class Tap:
    """把管理器的 `_log` 打点出来（不依赖环形缓冲，不丢早期行）。"""

    def __init__(self, mgr: KaiwuManager):
        self.lines: list[str] = []
        self.n = 0
        orig = mgr._log

        def _tee(line: str):
            self.lines.append(line)
            return orig(line)

        mgr._log = _tee          # 实例属性遮蔽绑定方法（调用处仍是 self._log）

    def dump(self) -> None:
        for l in self.lines[self.n:]:
            say("  | " + l)
        self.n = len(self.lines)


def cfg_disk() -> dict:
    return json.loads(config_path().read_text(encoding="utf-8"))


def key_of(cfg: dict, model: str) -> str:
    return KaiwuManager._ctx_cache_key(cfg, model)


def noted_tps(note: str) -> float | None:
    """从卡片备注里抠出「实测 X tok/s」（用于核对记录形态是否可复现）。"""
    m = re.search(r"实测\s*([\d.]+)\s*tok/s", note or "")
    return float(m.group(1)) if m else None


def want(section: str) -> bool:
    sel = [a for a in sys.argv[1:] if not a.startswith("-")]
    return (not sel) or any(s in " ".join(sel) for s in (section,))


# ------------------------------------------------------------------ 预检
CFG0 = load_config()
bak = config_path().parent / f"config.json.bak_{time.strftime('%Y%m%d')}_before_moeaccept"
if not bak.exists():
    bak.write_bytes(config_path().read_bytes())
    say(f"[备份] {bak.name}")

mgr = KaiwuManager()
tap = Tap(mgr)
say(f"[环境] llama-server = {resolve_llama_server(CFG0)}")
say(f"[环境] 老二进制 = {OLD_EXE}（存在={os.path.isfile(OLD_EXE)}）")
say(f"[环境] 总显存 = {mgr.total_vram_mib()} MiB")
say(f"[环境] gemma 挂载偏好 = {(CFG0.get('attach_overrides') or {}).get(GEMMA)}")
# ⚠️ DEFAULTS 会被 data/config.json 的显式值顶掉（2026-10-20 踩过：写死的
# 48/3 让 fit 扫描拿 48 token 比档位，噪声盖过 0.3 的档位差）
eff_tok = int(CFG0.get("ctx_tps_tokens") or 0)
eff_rnd = int(CFG0.get("ctx_tps_rounds") or 0)
eff_drp = int(CFG0.get("ctx_tps_drop") or 0)
say(f"[环境] 生效测速参数 = {eff_tok} token × {eff_rnd} 枪（丢前 {eff_drp}）"
    + ("" if (eff_tok >= 400 and eff_rnd >= 7 and eff_drp >= 2)
       else "  ⚠ 低于独立验收报告要求（400 token / 7 枪 / 丢 2）"))
verdict("预检：测速参数达报告要求（400 token / 7 枪 / 丢前 2）",
        eff_tok >= 400 and eff_rnd >= 7 and eff_drp >= 2,
        f"实际 {eff_tok}/{eff_rnd}/{eff_drp}")

# ------------------------------------------------------------------ §7.1
if want("7.1"):
    hdr("§7.1 gemma 完整测量（扫描选档 → 最终窗口复核 → 记录 fit/窗口）")
    t0 = time.time()
    res = mgr.measure_ctx(GEMMA, CFG0)
    tap.dump()
    say(f"[§7.1] best={res.get('best')} loads={res.get('loads')} 用时 {time.time()-t0:.0f}s")
    r1 = cfg_disk()
    k1 = key_of(r1, GEMMA)
    fit1 = (r1.get("fit_target") or {}).get(k1)
    verdict("§7.1 选出 fit 档位并落盘 fit_target", bool(fit1), f"{k1} → {fit1}")
    verdict("§7.1 同一记录不再写 moe_offload='cpu'（fit 优先）",
            (r1.get("moe_offload") or {}).get(k1) is None,
            f"moe_offload={r1.get('moe_offload')}")
    verdict("§7.1 记录窗口大于探测窗口", int(res.get("best") or 0) > int(CFG0.get("ctx_probe_ctx") or 32768),
            f"best={res.get('best')}")

    hdr("§7.1b 稳态复核（按记录形态重新加载；7 枪丢前 2、取后 5 的 mean/max）")
    cfg1 = load_config()
    ok1 = mgr.start(GEMMA, cfg1, timeout=300)
    tap.dump()
    cl1 = tap.lines[-1] if tap.lines else ""
    acct1 = mgr._parse_vram_log() or {}
    used1 = sum(float(acct1.get(k) or 0) for k in
                ("model_mib", "kv_mib", "compute_mib", "mmproj_mib"))
    mgr._measure_tps(cfg1, rounds=7, stat="mean")
    s = list(mgr._last_tps_samples)
    body = s[2:] or s
    mean = sum(body) / len(body) if body else 0.0
    mx = max(body) if body else 0.0
    say(f"[§7.1b] 加载={ok1} 授予 ctx={mgr.granted_ctx(cfg1)} 显存={used1:.0f} MiB"
        f"（{mgr._vram_pct(acct1):.0f}%）层={acct1.get('on_gpu')}/{acct1.get('total_layers')}")
    say(f"[§7.1b] 采样={s} → mean={mean:.2f} max={mx:.2f} tok/s")
    note1 = (r1.get("ctx_measured_note") or {}).get(k1) or ""
    nt = noted_tps(note1)
    say(f"[§7.1b] 记录备注里写的实测 = {nt} tok/s"
        + (f"（差 {abs(mx-nt)/nt*100:.0f}%）" if nt else "（没写）"))
    # 这三条合起来才是方案 §7.1b 的实质：**记录的形态本身**必须可复现 ——
    # 层全在 GPU、稳态达门槛、且与扫描时写进备注的实测数同量级。
    # 2026-10-20 的事故正是这里不过（记录 122880+fit2048 真跑掉 8 层、17.3 t/s）。
    verdict("§7.1b 记录形态下**层全在 GPU**（掉层 = 记录的窗口/档位不成立）",
            acct1.get("on_gpu") == acct1.get("total_layers")
            and bool(acct1.get("total_layers")),
            f"层 {acct1.get('on_gpu')}/{acct1.get('total_layers')}")
    verdict("§7.1b 稳态达 MoE 门槛",
            mx >= float(CFG0.get("ctx_moe_min_tps") or 15.0), f"max={mx:.2f} tok/s")
    verdict("§7.1b 稳态与备注实测同量级（≤35% 差）",
            bool(nt) and nt > 0 and abs(mx - nt) / nt <= 0.35,
            f"稳态 {mx:.2f} vs 备注 {nt}")
    say(f"[§7.1b] 记录形态命令行末段 = …{cl1[-150:]}")

# ------------------------------------------------------------------ §7.2-1
if want("7.2-1"):
    hdr("§7.2-1 稠密模型（Ornith）测量：与改前一致、无 fit_target")
    before = (CFG0.get("ctx_measured") or {}).get(key_of(CFG0, ORNITH))
    t0 = time.time()
    n0 = len(tap.lines)
    res2 = mgr.measure_ctx(ORNITH, load_config())
    seg2 = tap.lines[n0:]
    tap.dump()
    r2 = cfg_disk()
    k2 = key_of(r2, ORNITH)
    say(f"[§7.2-1] best={res2.get('best')}（改前记录 {before}）loads={res2.get('loads')}"
        f" 用时 {time.time()-t0:.0f}s")
    verdict("§7.2-1 稠密结论与改前一致", res2.get("best") == before,
            f"{res2.get('best')} vs {before}")
    verdict("§7.2-1 稠密模型不写 fit_target / moe_offload",
            (r2.get("fit_target") or {}).get(k2) is None
            and (r2.get("moe_offload") or {}).get(k2) is None)
    verdict("§7.2-1 稠密路径不出现扫档日志",
            not any("扫档" in l for l in seg2),
            f"本次段内扫档行数 {sum('扫档' in l for l in seg2)}")

# ------------------------------------------------------------------ §7.2-2
if want("7.2-2"):
    hdr("§7.2-2 老二进制：能力探测正确 + 不再被 --cors-origins 打死")
    if not os.path.isfile(OLD_EXE):
        verdict("§7.2-2 老二进制存在", False, "文件不在，跳过")
    else:
        cfg4 = dict(load_config())
        cfg4["llama_server_exe"] = OLD_EXE
        sup_fitt = mgr._llama_supports(cfg4, "-fitt")
        sup_cors = mgr._llama_supports(cfg4, "--cors-origins")
        say(f"[§7.2-2] 能力探测：-fitt={sup_fitt}  --cors-origins={sup_cors}")
        verdict("§7.2-2 探测走上 _spawn_env（老 exe 的 DLL 在 llama-deps）"
                "→ 支持 -fitt 不再被误判", sup_fitt is True)
        verdict("§7.2-2 老二进制确实不认 --cors-origins（故必须探测）",
                sup_cors is False)
        mgr.stop(cfg4)
        tap.n = len(tap.lines)
        ok4 = mgr.start(OLD_BIN_MODEL, dict(cfg4, last_model=OLD_BIN_MODEL),
                        ctx_override=8192, timeout=300)
        tap.dump()
        cl4 = next((l[2:] for l in reversed(tap.lines) if l.startswith("$ ")), "")
        seg = "".join(tap.lines[tap.n:])
        say(f"[§7.2-2] 用 {OLD_BIN_MODEL}（qwen35，老构建认识）加载={ok4}"
            f" err={mgr.last_error[:100]!r}")
        say(f"[§7.2-2] 命令末段 = …{cl4[-150:]}")
        verdict("§7.2-2 老二进制能起来（不再被 --cors-origins 打死）", ok4)
        verdict("§7.2-2 命令里没有 --cors-origins，也没有 invalid argument/"
                "unrecognized option",
                "--cors-origins" not in cl4 and "invalid argument" not in seg
                and "unrecognized option" not in seg)

# ------------------------------------------------------------------ §7.2-3
if want("7.2-3"):
    hdr("§7.2-3 手动 ctx=4096 加载：不带 --cpu-moe、带记录里的 -fitt")
    r3 = cfg_disk()
    k3 = key_of(r3, GEMMA)
    fit3 = (r3.get("fit_target") or {}).get(k3)
    tap.n = len(tap.lines)
    ok3 = mgr.start(GEMMA, load_config(), ctx_override=4096, timeout=300)
    tap.dump()
    cl3 = next((l[2:] for l in reversed(tap.lines) if l.startswith("$ ")), "")
    acct3 = mgr._parse_vram_log() or {}
    np3 = int(acct3.get("n_part") or 0)
    tot3 = int(acct3.get("total_layers") or 0)
    say(f"[§7.2-3] 加载={ok3} 命令末段 = …{cl3[-150:]}")
    say(f"[§7.2-3] n_part={np3} 总层={tot3} → 专家层在显存 {max(tot3-np3, 0)}")
    verdict("§7.2-3 手动窗口也走 fit（带 -fitt）", fit3 and f"-fitt {fit3}" in cl3)
    verdict("§7.2-3 手动窗口不再带 --cpu-moe", "--cpu-moe" not in cl3)

# ------------------------------------------------------------------ §7.2-4
if want("7.2-4"):
    hdr("§7.2-4 换 KV 量化 → 指纹失配 → 自动重测（真实配置，不烧 GPU）")
    k5 = key_of(load_config(), GEMMA)
    cfg5 = dict(load_config())
    cfg5["cache_type_k"] = "f16"
    cfg5["cache_type_v"] = "f16"
    q_ok = mgr._ctx_cache_valid(load_config(), k5, GEMMA)
    q_me = mgr._should_measure(load_config(), GEMMA)
    f_ok = mgr._ctx_cache_valid(cfg5, k5, GEMMA)
    f_me = mgr._should_measure(cfg5, GEMMA)
    say(f"[§7.2-4] q8_0：可信={q_ok} 需重测={q_me}；f16：可信={f_ok} 需重测={f_me}")
    verdict("§7.2-4 q8_0 下缓存有效、不重测", q_ok is True and q_me is False)
    verdict("§7.2-4 f16 下指纹失配 → 自动重测", f_ok is False and f_me is True)

# ------------------------------------------------------------------ §7.2-5
if want("7.2-5"):
    hdr("§7.2-5 并发加载守卫：加载中再点一次 → 毫秒级拒绝")
    cfgn = load_config()
    mgr.stop(cfgn)
    th = threading.Thread(target=lambda: mgr.start(GEMMA, cfgn, ctx_override=65536,
                                                   timeout=300), daemon=True)
    th.start()
    time.sleep(0.6)
    t0 = time.time()
    ok6 = mgr.start(GEMMA, cfgn, ctx_override=4096)
    dt = (time.time() - t0) * 1000
    say(f"[§7.2-5] 第二次点击 → {ok6} 耗时 {dt:.0f} ms err={mgr.last_error!r}")
    th.join()
    tap.dump()
    verdict("§7.2-5 并发第二次点击被拒且是毫秒级",
            ok6 is False and dt < 200, f"{dt:.0f} ms")

# ------------------------------------------------------------------ 收尾
hdr("收尾：卸载模型 + 配置差异")
mgr.stop(load_config())
after = cfg_disk()
diff = {k: {"before": CFG0.get(k), "after": after.get(k)}
        for k in sorted(set(CFG0) | set(after)) if CFG0.get(k) != after.get(k)}
say(json.dumps(diff, ensure_ascii=False, indent=2))

bad = [n for n, p, _ in VERDICTS if not p]
say("\n" + "=" * 74)
say(f"验收判定：{len(VERDICTS) - len(bad)}/{len(VERDICTS)} 通过")
for n, p, d in VERDICTS:
    say(f"  {'PASS' if p else 'FAIL'}  {n}" + (f" —— {d}" if d else ""))
LOGP.write_text("\n".join(tap.lines), encoding="utf-8")
say(f"\n完整 llama-server 日志：{LOGP}")
sys.exit(1 if bad else 0)
