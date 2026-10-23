# -*- coding: utf-8 -*-
"""fit 档位（`-fitt`）的**先算后测**：预测 n_part、规划只测哪几档。

背景（2026-10-21 真机实测，见 tests/_test_fit_timing.py + memorix #109/#105）：
扫档是 kaiwu 最贵的一步 —— 每档 = 一次真加载（10~20s）+ 7 枪测速（60~100s），
LFM2.5 上扫 7 档要 **606 秒 = 10.1 分钟**，用户就在那儿干等。而实测发现：
**档位 → 专家切分（n_part）是可以先算出来的**，两个结构迥异的 MoE 模型
（gemma-4-26B-A4B 31 层 / LFM2.5-24B-A2B 41 层）共 **12/12 零误差**。

原理：llama.cpp 的 fit 在加载时会把「每个候选切分占多少显存」打进 trace 日志：

    id=0, n_layer=41, n_part=26, overflow_type=4, mem= 6213 MiB

它挑档的方式是「把层逐个从『只有稠密权重在显存』升级成『整层专家也在显存』，
每升一层 n_part 减 1，直到装不下」⇒ 预测公式就是：

    预算(fitt) = 运行时空闲显存 − fitt
    预测 n_part = 使 mem(n_part) ≤ 预算 的**最小** n_part

n_part 越小 = 专家越多在显存 = 越快。所以「预测 n_part 最小的那档」就是
「预测最快的那档」，只需要再测它和左右各一邻档来裁判（数学只用来定位，
实测才是裁判 —— 实测闭环是护城河）。

⚠️ 三个必须避开的坑（都踩过）：
  ① 阶梯要**按 n_layer 过滤**：日志里有一行 `n_layer=0, n_part=0`（"把所有
     MoE tensor 搬到系统内存"的探测），混进来会让预测全部退化成 0（实测 0/7）。
  ② 日志里的 `id=0, target=NNNN MiB` **不可信**（各档分别打 5920/4896/6432/6688，
     与 free−fitt 算的 2848/4896/6432/6688 对不上）→ 预算一律自己算。
  ③ n_part **依赖 ctx**（LFM fitt512：@118784 → 26、@32768 → 21）→ 预测只能用
     **同窗口**加载出来的阶梯，跨窗口套用必错。
"""
from __future__ import annotations

import re
from typing import Iterable

# fit trace 里的候选行：`id=0, n_layer=41, n_part=26, overflow_type=4, mem= 6213 MiB`
# overflow_type 是"哪类张量溢出去了"（2=UP / 3=GATE / 4=全部），只影响 mem 的小数，
# 不影响 n_part，所以只取 n_part 与 mem，同一 n_part 取 mem 最小的那条。
_RE_CAND = re.compile(
    r"id\s*=\s*\d+,\s*n_layer=\s*(\d+),\s*n_part=\s*(\d+),"
    r"\s*overflow_type=\s*\d+,\s*mem=\s*(\d+)\s*MiB")
# 运行时空闲显存（含桌面/浏览器占用，**不是**总显存）
_RE_FREE = re.compile(r"vs\.\s*(\d+)\s*MiB of free device memory")
# 每次加载的 fit 阶段开头（用它切出「最近一次加载」的日志段）
_MARK_FIT = "fitting params to device memory"


def last_fit_section(log: str) -> str:
    """切出**最近一次加载**的 fit 日志段。

    self._logbuf 跨加载累积（deque 不清空），整段拿去解析会把上一次加载的
    空闲显存当成这次的 —— 显存会随桌面占用变，预算一错预测就全错。
    """
    idx = log.rfind(_MARK_FIT)
    return log[idx:] if idx >= 0 else log


def parse_fit_ladder(log: str) -> dict:
    """解析「切分阶梯」+ 运行时空闲显存。

    返回 {"free_mib": int|None, "ladder": {n_part: mem_mib}, "n_layer": int|None}
      · ladder 只含**总层数那一组**候选（n_layer=0 的探针行被滤掉，坑①）；
      · 同一 n_part 多行时取 mem 最小（不同 overflow_type）；
      · 最后强制单调：n_part 升序时 mem 必须非升（违反的点直接丢掉 —— 兜底
        防止别的探测行混进来把曲线弄歪）。

    解析不出（老版本没这些 trace 行 / 日志被截断）返回空 ladder，
    调用方据此退回**全量扫描**（旧行为，不会更差）。
    """
    seg = last_fit_section(log or "")
    rows: list[tuple[int, int, int]] = []
    for n_layer, n_part, mem in _RE_CAND.findall(seg):
        rows.append((int(n_layer), int(n_part), int(mem)))
    m_free = _RE_FREE.search(seg)
    free_mib = int(m_free.group(1)) if m_free else None
    top = max((r[0] for r in rows), default=0)
    ladder: dict[int, int] = {}
    if top > 0:
        for n_layer, n_part, mem in rows:
            if n_layer != top:
                continue
            prev = ladder.get(n_part)
            if prev is None or mem < prev:
                ladder[n_part] = mem
    clean: dict[int, int] = {}
    last = None
    for n in sorted(ladder):
        mem = ladder[n]
        if last is not None and mem > last:
            continue                       # 与"mem 随 n_part 递减"矛盾 → 丢掉
        clean[n] = mem
        last = mem
    return {"free_mib": free_mib, "ladder": clean, "n_layer": top or None}


def merge_ladder(a: dict, b: dict) -> dict:
    """两阶梯合并（同 n_part 取 min mem）。

    为什么要合并：阶梯**随目标变大而变丰富** —— fitt 4096 那次只试了 5 个点，
    fitt 64 那次试了 8 个点。

    ⚠️ 当前**不在生产路径上**：扫档计划在开始前一次性算出（用探测那次加载
    的阶梯），实测不改计划 —— 改计划会打乱「只测哪几档」的可解释性。
    保留这个函数是因为它把「阶梯会变丰富」这件事表达清楚了，且规格测试
    锁住它的行为；将来若要做「边测边重规划」时直接用（2026-10-21 验收时
    发现原先的合并调用是死代码，已移除，避免文档宣称「越测越准」而代码
    根本没读合并结果）。
    """
    out = dict(a or {})
    for n, mem in (b or {}).items():
        prev = out.get(n)
        if prev is None or mem < prev:
            out[n] = int(mem)
    return out


def mem_at(ladder: dict, n: float) -> float:
    """阶梯上的分段线性取值；两端线性外推。

    mem 随 n_part **单调递减**（专家越少留在 CPU，占用越大）。
    线性外推在本机两个模型上都成立：LFM 每层专家 ≈ 249 MiB 且严格等距
    （27:5964 → 26:6213 → 25:6462 …），gemma 也一致。
    """
    ns = sorted(int(k) for k in ladder)
    if not ns:
        return 0.0
    get = lambda k: float(ladder[k] if k in ladder else ladder[str(k)])
    if len(ns) == 1:
        return get(ns[0])
    if n <= ns[0]:
        a, b = ns[0], ns[1]
    elif n >= ns[-1]:
        a, b = ns[-2], ns[-1]
    else:
        a = b = ns[0]
        for i in range(len(ns) - 1):
            if ns[i] <= n <= ns[i + 1]:
                a, b = ns[i], ns[i + 1]
                break
    if b == a:
        return get(a)
    return get(a) + (get(b) - get(a)) * (n - a) / (b - a)


def predict_n_part(ladder: dict, free_mib: int | None, fitt: int,
                   n_layer: int | None = None) -> tuple[int | None, str]:
    """给定档位 → 预测 n_part。返回 (n_part, 说明)；None = 连最省都装不下。

    「层级别回填」路径（pred=None）意味着稠密层都放不满、必然更慢 —— 调用方
    应当跳过它，这也是保守档（4096/2048）上最大的省钱点（gemma 那两个档
    白占 323 秒 = 扫档总时的 39%）。
    """
    if not ladder or not free_mib:
        return None, "没有阶梯或空闲显存"
    target = int(free_mib) - int(fitt)
    hi = max(max(int(k) for k in ladder), int(n_layer or 0))
    for n in range(0, hi + 1):
        if mem_at(ladder, n) <= target:
            return n, (f"预算 {target} MiB（空闲 {free_mib} − 预留 {fitt}），"
                       f"mem(n_part={n})={mem_at(ladder, n):.0f} ≤ 预算")
    return None, (f"预算 {target} MiB 连最省配置（n_part={hi} 需 "
                  f"{mem_at(ladder, hi):.0f} MiB）都装不下 → 层级别回填")


def predict_vram(ladder: dict, n_part: int) -> float:
    """预测该切分下占多少显存（MiB）。与实测差 1~3%（不含计算缓冲的零头），
    只用来提前挡掉明显贴顶的档，**实测水位仍是裁判**。"""
    return mem_at(ladder, n_part)


def plan_scan(ladder: dict, free_mib: int | None, candidates: Iterable[int],
              n_layer: int | None = None, total_vram_mib: int | None = None,
              watermark_pct: float = 88.0) -> dict:
    """规划**只测哪几档**：预测最快的一档 + 左右各一邻档。

    candidates 必须按「保守 → 激进」排好（与 _FIT_CANDIDATES 同序）。
    返回 {"plan": [...], "skipped": {fitt: 原因}, "reason": ""}：
      · plan 按 candidates 顺序排列，每项 {"fitt","n_part","mem","pct","why"}；
      · plan 为空 = 预测不可用（没阶梯/没空闲显存/阶梯太浅）→ 调用方退回
        **全量扫描**（旧行为，不会更差）。这时 reason 说明原因。

    规则（用户 2026-10-21 拍板）：
      ① 预测 n_part 最小 = 预测最快；同 n_part 取**更保守**那档（fitt 更大）。
      ② 左右各取一邻档做裁判；**与已选档 n_part 相同的直接跳过**（同一配置，
         分不出速度差 —— 实测 LFM 128/64 差 0.56 < 噪声 1.2）。
      ③ 邻档被跳过后继续往外走一格补齐（最多补 2 格），保证"更保守"那一侧
         总有一档可比 —— 选档规则（同分取保守档）要靠它判是否在噪声内。
      ④ 预测显存超水位（默认 88%）的档不进计划（贴顶会被 compute buffer 挤，
         实测反而更慢）；预测不出 n_part 的（层级别回填）也不进 —— 后者正是
         gemma 那两个保守档，占了扫档总时的 39%，是最大的一笔省钱。
    """
    cs = [int(c) for c in candidates]
    skipped: dict[int, str] = {}
    if not ladder or not free_mib or not cs:
        return {"plan": [], "skipped": skipped, "preds": {},
                "reason": "日志里没有切分阶梯或空闲显存（老版本二进制？）"}
    preds: dict[int, dict] = {}
    all_preds: dict[int, int | None] = {}   # 每个候选档的预测 n_part（含被跳过的）
    for i, fitt in enumerate(cs):
        n, why = predict_n_part(ladder, free_mib, fitt, n_layer)
        all_preds[fitt] = n
        if n is None:
            skipped[fitt] = "预测为层级别回填（稠密层都放不满，必然更慢）"
            continue
        mem = predict_vram(ladder, n)
        pct = (mem / total_vram_mib * 100.0) if total_vram_mib else None
        if pct is not None and pct > watermark_pct:
            skipped[fitt] = f"预测显存 {mem:.0f} MiB（{pct:.0f}%）超水位 {watermark_pct:.0f}%"
            continue
        preds[i] = {"fitt": fitt, "n_part": n, "mem": round(mem, 1),
                    "pct": round(pct, 1) if pct is not None else None, "why": why}
    if not preds:
        return {"plan": [], "skipped": skipped, "preds": all_preds,
                "reason": "所有候选档都被预测判为不可用（贴顶/层级别回填）"}
    def leftmost_same(i: int) -> int:
        """同一 n_part 的一串档里，取**最保守**那个（fitt 最大）。

        同一 n_part = 同一套专家分配 = 速度分不出来，那就没理由多占显存。
        真机教训（gemma @32768）：预测 n_part 是 512:22 / 256:21 / 128:21 / 64:20，
        若把「64 的邻档」原样取成 128，就会拿 82% 水位去换一个噪声内的读数；
        归一到 256（同为 n_part=21、水位低 2pp）才是该测的那个。

        被归一掉（跳过）的中间档要记进 skipped —— 否则用户从日志里看不到
        「为什么这一档没测」，会以为漏测（2026-10-21 验收发现）。
        """
        start = i
        while i - 1 in preds and preds[i - 1]["n_part"] == preds[i]["n_part"]:
            i -= 1
            skipped[cs[i]] = (f"与 {cs[start]} 同 n_part={preds[start]['n_part']}"
                              "（同一配置，分不出速度差）→ 归一到更保守的那档")
        return i

    # ① 预测最快：n_part 最小；同 n_part 取 fitt 更大（更保守）的
    best_i = min(preds, key=lambda i: (preds[i]["n_part"], -cs[i]))
    best_i = leftmost_same(best_i)
    best_n = preds[best_i]["n_part"]
    chosen = {best_i}
    # ②③ 左右各一邻档，跳过同 n_part，最多往外补 2 格
    for step in (-1, 1):
        i, hops = best_i + step, 0
        while 0 <= i < len(cs) and hops < 2:
            if i in preds:
                if preds[i]["n_part"] != best_n:
                    chosen.add(leftmost_same(i))
                    break
                skipped[cs[i]] = f"与 {cs[best_i]} 同 n_part={best_n}（同一配置，分不出速度差）"
            i += step
            hops += 1
    plan = [preds[i] for i in sorted(chosen)]
    for p in plan:
        skipped.pop(p["fitt"], None)
    # 剩下的档也要有交代：既不是「预测最快」也不是它的邻档 → 不测。
    # **每一档都必须能在日志里找到为什么没测**（不然读日志的人会以为漏测）。
    chosen_fitts = {p["fitt"] for p in plan}
    for fitt, n in all_preds.items():
        if fitt in chosen_fitts or fitt in skipped:
            continue
        if n is None:
            skipped[fitt] = "预测为层级别回填（稠密层都放不满，必然更慢）"
        else:
            skipped[fitt] = (f"预测 n_part={n}，不是预测最快档（n_part="
                             f"{best_n}）也不是它的邻档 → 不测")
    # all_preds 含**每个**候选档的预测 n_part（连被跳过的）—— 调用方要在扫档后
    # 用**更丰富的阶梯**重算一遍：如果某档的预测变了，说明当初「跳过它」的
    # 判断是基于不够准的阶梯做的，可能错判 → 该补测那档。
    return {"plan": plan, "skipped": skipped, "preds": all_preds, "reason": ""}
