"""模型目录扫描：把用户给的一个文件夹解析成「模型单元」。

2026-10-15 定下的目录约定（「一模型一文件夹」）—— 给别人的机器用时，
模型清单不写死，全靠扫目录；伴生文件（视觉塔 mmproj / MTP 头 mtp-*）
的归属也不能靠文件名猜，靠**文件夹边界**定：

    模型库根目录/                       ← 「添加模型文件夹」选这个
    ├─ gemma-4-26B-A4B-it-ud/          ← 一个子文件夹 = 一个模型
    │  ├─ gemma-4-26B-A4B-it-ud-IQ3_S.gguf    主权重（可多个量化变体）
    │  ├─ mmproj-gemma-4-26B-BF16.gguf        视觉塔（可选）
    │  └─ mtp-gemma-4-26B-A4B-it-Q8_0.gguf    MTP 头（可选）
    ├─ Qwen3.5-4B-UD/
    │  ├─ Qwen3.5-4B-UD-Q5_K_XL.gguf
    │  └─ Qwen3.5-4B-UD-Q4_K_M.gguf           同底座不同量化 = 变体
    └─ josie-4b-instruct-32k/
       └─ josie-4b-instruct-32k.gguf

配对规则（kaiwu_manager 加载时同样遵守，见 _pick_sidecar）：
  1. 伴生文件名里含该模型的**匹配键**（底座名去掉 -ud/-it 等后缀）→ 自证，挂；
  2. 文件夹里只有这一个模型（只有一组主权重）→ 整个文件夹都是它的，挂；
  3. 其余 → 不猜。宁可不挂让用户在卡片上手选，也不挂错。

旧布局兼容：.gguf 直接平放在根目录时，逐个登记（名字用完整文件名），
伴生文件按上面的 1/3 规则来（扁平目录多模型混放时判定不了归属）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# 扫描时不进的目录（缓存/依赖/构建产物，与 local_fs.IGNORE_DIRS 同思路，
# 这里刻意不 import 那个模块 —— 扫描逻辑保持零依赖，便于单测）
_SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv",
              "venv", "env", ".idea", ".vscode", ".cache", "dist", "build"}

# 量化后缀：只剥**尾部**的量化档，且必须带 -/_ 分隔，避免误伤模型名里的数字
# （"josie-4b-instruct-32k" 的 32k 不是量化；"gemma-4-26B" 的 26B 也不是）
_QUANT_TAIL = re.compile(
    r"[-_]?(?:"
    r"[Ii][Qq]\d+[_.]?[A-Za-z0-9]*"       # IQ3_S / IQ4_XS / IQ2_XXS / IQ4_NL
    r"|[Tt][Qq]\d+[_.]\d+"                # TQ1_0 / TQ2_0
    r"|[Qq]\d+(?:[_.][A-Za-z0-9]+)*"      # Q4_K_M / Q5_K_XL / Q8_0 / Q6_K / Q4_0
    r"|BF16|F16|F32|FP16|FP8|MXFP4"       # 非 K 量化档（大小写不敏感由 re.I 覆盖）
    r")$", re.I)

# 视觉塔量化偏好：私有文件夹里出现多个塔时按质量挑（纯文本质量
# BF16 > F16 > Q8_0 > …），确定性排序，不靠字典序碰运气
_TOWER_QUANT_PREF = ("bf16", "f16", "q8_0", "q6_k", "q5_k_m", "q5_k_s",
                     "q5_0", "q4_k_m", "q4_k_s", "q4_0", "q3", "q2", "iq")


@dataclass
class ModelUnit:
    """一个模型单元 = 一个装着主权重（+伴生文件）的文件夹。"""
    folder: Path
    # 底座名 → 主权重文件列表（同底座不同量化 = 变体，≥2 个时 UI 出下拉）
    groups: dict[str, list[Path]] = field(default_factory=dict)
    towers: list[Path] = field(default_factory=list)   # mmproj*.gguf
    drafts: list[Path] = field(default_factory=list)   # mtp-*.gguf / mtp.*.gguf

    @property
    def private(self) -> bool:
        """整个文件夹是否只属于一个模型（只有一组主权重）。"""
        return len(self.groups) == 1


def base_stem(stem: str) -> str:
    """剥掉尾部量化档，返回底座名。

    'Qwen3.5-4B-UD-Q5_K_XL' → 'Qwen3.5-4B-UD'
    'gemma-4-26B-A4B-it-ud-IQ3_S' → 'gemma-4-26B-A4B-it-ud'
    'josie-4b-instruct-32k' → 'josie-4b-instruct-32k'（无量化后缀，原样）
    剥空了（文件名整个就是量化档）就返回原串，绝不返回空名。
    """
    s = (stem or "").strip()
    if not s:
        return s
    b = _QUANT_TAIL.sub("", s).strip("-_ .")
    return b or s


def quant_label(stem: str) -> str:
    """主权重文件名的量化档标签：'Qwen3.5-4B-UD-Q5_K_XL' → 'Q5_K_XL'。
    没有量化后缀返回 ''（调用方自行决定显示什么）。"""
    s = (stem or "").strip()
    b = base_stem(s)
    if not b or b == s:
        return ""
    return s[len(b):].lstrip("-_") or ""


def is_sidecar(name: str) -> bool:
    """是不是伴生文件（视觉塔 / MTP 头）—— 不能单独加载，别列进模型清单。"""
    low = (name or "").lower()
    return low.startswith(("mmproj", "mtp-", "mtp."))


def match_keys(base: str) -> list[str]:
    """底座名 → 用于「文件名自证归属」的匹配键，**长的优先**。

    模型 tag 常带一长串修饰（gemma-4-26B-A4B-it-ud），而塔文件名通常只写到
    gemma-4-26B —— 用完整底座名做包含匹配会全部落空，所以按「-」分段从长到短
    生成键梯度：先试全名，再依次砍尾部段（-ud / -it / -a4b…），最短到两段、
    总长 ≥6 为止（再短就容易张冠李戴，比如 qwen3.5 同时是 4B 和 9B 的前缀）。
    调用方按顺序取第一个「恰好命中一个候选」的键，长短互补且不误判。
    """
    b = (base or "").strip().lower()
    if not b:
        return []
    parts = [p for p in b.split("-") if p]
    out: list[str] = []
    for n in range(len(parts), 1, -1):          # 全名 → … → 两段
        k = "-".join(parts[:n])
        if len(k) >= 6 and k not in out:
            out.append(k)
    if b not in out:
        out.append(b)                            # 单段名（如 ornith）也试一次全串
    return out


def scan_units(root, max_depth: int = 3) -> list[ModelUnit]:
    """递归扫描 root，返回所有「模型单元」（含主权重的目录）。

    - 任意层级、只要目录里出现主权重 .gguf 就算一个单元（多深都扫，
      但默认最多向下 3 层，防止用户误选整个硬盘根目录）；
    - 根目录自己有权重也算一个单元（folder == root）—— 调用方据此
      区分「新布局（子文件夹）」和「旧布局（平铺）」；
    - 隐藏目录与缓存/构建目录整棵跳过。
    """
    root = Path(root)
    units: list[ModelUnit] = []

    def walk(d: Path, depth: int) -> None:
        mains: list[Path] = []
        towers: list[Path] = []
        drafts: list[Path] = []
        subdirs: list[Path] = []
        try:
            entries = list(d.iterdir())
        except OSError:
            return
        for e in entries:
            try:
                if e.is_dir():
                    if depth < max_depth and not e.name.startswith(".") \
                            and e.name.lower() not in _SKIP_DIRS:
                        subdirs.append(e)
                    continue
                if e.suffix.lower() != ".gguf":
                    continue
            except OSError:
                continue
            low = e.name.lower()
            if low.startswith("mmproj"):
                towers.append(e)
            elif low.startswith(("mtp-", "mtp.")):
                drafts.append(e)
            else:
                mains.append(e)
        if mains:
            groups: dict[str, list[Path]] = {}
            for f in sorted(mains, key=lambda p: p.name.lower()):
                groups.setdefault(base_stem(f.stem), []).append(f)
            units.append(ModelUnit(folder=d, groups=groups,
                                   towers=sorted(towers, key=lambda p: p.name.lower()),
                                   drafts=sorted(drafts, key=lambda p: p.name.lower())))
        for sd in sorted(subdirs, key=lambda p: p.name.lower()):
            walk(sd, depth + 1)

    walk(root, 0)
    return units


def _tower_rank(p: Path) -> tuple:
    """视觉塔量化偏好排序键（越小越优先）；未知档位排最后按名字兜底。"""
    low = p.stem.lower()
    for i, q in enumerate(_TOWER_QUANT_PREF):
        if q in low:
            return (i, p.name.lower())
    return (len(_TOWER_QUANT_PREF), p.name.lower())


def bind_sidecars(unit: ModelUnit, base: str) -> dict:
    """给单元里的一个模型（base）定伴生文件归属。

    返回 {"vision": (Path|None, how), "draft": (Path|None, how)}，
    how ∈ "name"（文件名自证）/ "folder"（文件夹私有整包归属）/ ""（没配上）。

    规则与加载时（kaiwu_manager._pick_sidecar）一致，宁可返回 None
    也不猜 —— 配不上的模型，卡片上会给「待选择」让用户手动指定。
    """
    out = {"vision": (None, ""), "draft": (None, "")}
    keys = match_keys(base)
    for kind, cands in (("vision", unit.towers), ("draft", unit.drafts)):
        if not cands:
            continue
        hit: Path | None = None
        # ① 文件名里含匹配键 → 自证（多组共存的文件夹里也能逐个认出来）
        for key in keys:
            hits = [c for c in cands if key in c.name.lower()]
            if len(hits) == 1:
                hit = hits[0]
                out[kind] = (hit, "name")
                break
        if hit is not None:
            continue
        # ② 文件夹只属于这一个模型 → 同目录的伴生文件必然是它的
        if unit.private:
            pick = (sorted(cands, key=_tower_rank)[0]
                    if kind == "vision" else sorted(cands, key=lambda p: p.name.lower())[0])
            out[kind] = (pick, "folder")
        # ③ 多组共存又互不自证 → 不猜，留给 UI 手选
    return out


def same_dir(a, b) -> bool:
    """两个路径是否同一个目录（Windows 大小写不敏感）。"""
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a).lower() == str(b).lower()
