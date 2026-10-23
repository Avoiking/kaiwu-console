"""模型思考控制能力探测 —— **自动识别，加新模型不用改代码**。

## 它解决什么

不是所有模型都能调思考强度，而且**旋钮名字各家不同**：

| 模型 | 模板里的变量 | 能调什么 | UI |
|---|---|---|---|
| gpt-oss-20b | `reasoning_effort` | low / medium / high | 三档 |
| Qwen3.5-4B / Qwen3 | `enable_thinking` | true / false | 开 / 关 |
| josie-4b-* | 无 | 不能调 | 控件隐藏 |
| LFM2.5 | `preserve_thinking`（只决定"要不要保留历史思考"，不是档位） | 不能调 | 控件隐藏 |

写死两个名字迟早会被新模型打脸，所以这里做**通用判定**：扫模板里用到的外部变量，
按"名字里有没有思考相关词 + 它被怎么用"来分类。

## 判定逻辑（都是可读的启发式，不猜玄学）

1. **收集外部变量**：只认这几种强信号 —— `X is (not) defined`、`X | default(...)`、
   `{% if X %}` / `{% set X = ... %}`、`X == "字面量"`。
   `message.thinking` 这类属性访问不算（正则排除点号前缀）。
2. **按名字筛**：名字里含 reason / think / effort / level / budget / intensity 才算候选。
3. **排掉"不是档位"的**：`preserve_*` / `include_*` / `keep_*` / `hide_*` 前缀
   —— 它们控制的是"要不要把思考文本留下来"，不是"想多久"。
4. **分类**：
   - 名字含 effort / level / budget / intensity → **effort**（三档：low/medium/high；
     若模板里能查到字面量比较就用查到的）
   - 否则 → **switch**（true/false）
   - 一个候选都没有 → **none**

## 用法

    python -m app.model_caps              # 列出本地所有模型的能力
    python -m app.model_caps gpt-oss-20b  # 看某个模型的判定细节与证据
    python -m app.model_caps <某个.gguf 路径>

**不加载模型**：只读 GGUF 的元数据段，11GB 的模型也只要 0.35 秒，
而且找到 `tokenizer.chat_template` 就早退（不读后面几十 MB 的词表）。
"""
from __future__ import annotations

import re
import struct
import sys
import threading
from pathlib import Path

from .kaiwu_manager import KaiwuManager

MAX_READ = 16 * 1024 * 1024   # 元数据最多读这么多就放弃（防畸形文件拖住界面）

_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
        7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_TPL_KEY = "tokenizer.chat_template"

# think 取值："effort"（三档）/ "switch"（开关）/ "none"（没有可调项，UI 藏起来）
KINDS = ("effort", "switch", "none")

# 名字里带这些词才算"思考控制"的候选
_WORDS = ("reason", "think", "effort", "level", "budget", "intensity")
# 但这些前缀说明它控制的是"思考文本要不要留着"，不是"想多久" —— 不算候选
_SKIP_PREFIX = ("preserve_", "include_", "keep_", "hide_", "show_", "print_")
# effort 类的典型可选值（模板里查不到字面量比较时的默认）
EFFORT_VALUES = ("low", "medium", "high")

_cache: dict[tuple, dict] = {}
_lock = threading.Lock()


# ============================================================ GGUF 读取
def _read_value(f, t: int, budget: list[int]):
    data = f.read(_SZ[t]) if t in _SZ else b""
    budget[0] += len(data)
    if budget[0] > MAX_READ:
        raise ValueError("元数据过大，放弃")
    if t in _FMT:
        return struct.unpack(_FMT[t], data)[0]
    if t == 8:                                   # 字符串
        n = struct.unpack("<Q", f.read(8))[0]
        budget[0] += 8 + n
        if budget[0] > MAX_READ:
            raise ValueError("元数据过大，放弃")
        return f.read(n).decode("utf-8", "replace")
    if t == 9:                                   # 数组
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        budget[0] += 12
        return [_read_value(f, et, budget) for _ in range(n)]
    raise ValueError(f"未知类型 {t}")


def read_chat_template(path: str, stats: dict | None = None) -> str:
    """只读 GGUF 元数据段里的 `tokenizer.chat_template`；没有就返回 ""。

    stats 可选：传个 dict 进去，会填上 {"bytes": 实际读了多少字节, "found": 有没有找到}。
    加这个是为了能拿数字说话 —— 探测的代价到底多大（通常是几百 KB 到几 MB 的文件头）。
    """
    budget = [0]
    if isinstance(stats, dict):
        stats.clear()
        stats.update({"bytes": 0, "found": False})
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return ""
            f.read(4)                                # version
            f.read(8)                                # tensor_count
            n_kv = struct.unpack("<Q", f.read(8))[0]
            budget[0] += 24
            for _ in range(n_kv):
                klen = struct.unpack("<Q", f.read(8))[0]
                budget[0] += 8 + klen
                if budget[0] > MAX_READ:
                    return ""
                key = f.read(klen).decode("utf-8", "replace")
                vt = struct.unpack("<I", f.read(4))[0]
                val = _read_value(f, vt, budget)
                if key == _TPL_KEY:
                    if isinstance(stats, dict):
                        stats.update({"bytes": budget[0], "found": True})
                    return val if isinstance(val, str) else ""
    except (OSError, ValueError, struct.error):
        return ""
    finally:
        if isinstance(stats, dict) and not stats.get("found"):
            stats["bytes"] = budget[0]
    return ""


# ============================================================ 判定
_BLOCK = re.compile(r"\{\{(.*?)\}\}|\{%(.*?)%\}", re.S)
_NAME = re.compile(r"(?<![.\w])([A-Za-z_]\w*)")
_BUILTIN = {
    "messages", "tools", "add_generation_prompt", "loop", "namespace",
    "raise_exception", "strftime_now", "message", "item", "content", "role",
    "true", "false", "none", "null", "and", "or", "not", "if", "else", "elif",
    "endif", "for", "in", "endfor", "set", "endset", "string", "mapping",
    "iterable", "defined", "undefined", "length", "int", "float", "trim",
    "default", "system", "user", "assistant", "tool", "text", "image",
    "image_url", "video", "type", "sender", "functions", "function", "call",
}
# 变量出现的强信号（出现这些才算"外部可传的参数"）。
# ⚠️ 这些正则都作用在**块内文本**上（`{% %}` 之间的那截，不含 `{%`），所以
#    第三条不能写 `\{%\s*if` —— 实测就是这么漏掉最朴素的 `{% if X %}` 形式的。
# ⚠️ **不收 `{% set X = ... %}`**：那是模板在**定义局部变量**，不是读外部参数。
#    实测把 josie-4b-thinking 的 `{% set reasoning_content = '' %}` 当成了"思考开关"。
# ⚠️ 每条都要排除"点号前缀"：`message.thinking is defined` 里的 thinking 是**消息字段**，
#    不是模板参数 —— 不排除的话 LFM2.5 会被误判成"有 thinking 开关"（实测踩过）。
_SIGNALS = (
    re.compile(r"(?<![.\w])(\w+)\s+is\s+(?:not\s+)?defined"),
    re.compile(r"(?<![.\w])(\w+)\s*\|\s*default"),
    re.compile(r"^\s*-?\s*(?:if|elif)\s+(\w+)"),
    re.compile(r"(?<![.\w])(\w+)\s*(?:==|!=)\s*[\"']"),
)
# 名字以这些结尾的多半是"装思考文本的容器"，不是旋钮
_SKIP_SUFFIX = ("_content", "_text", "_str", "_string", "_message")


def _candidates(template: str) -> dict[str, str]:
    """外部变量名 → 证据片段（只收思考相关的候选）。"""
    template = template or ""
    # 凡是模板里以 `message.X` 形式出现过的名字，都是**消息字段**，不是模板参数
    msg_fields = set(re.findall(r"message\.(\w+)", template))
    found: dict[str, str] = {}
    for m in _BLOCK.finditer(template):
        block = (m.group(1) or m.group(2) or "").strip()
        if not block:
            continue
        for sig in _SIGNALS:
            for mm in sig.finditer(block):
                name = mm.group(1)
                low = name.lower()
                if name in _BUILTIN or name in msg_fields:
                    continue
                if name.startswith(_SKIP_PREFIX) or name.endswith(_SKIP_SUFFIX):
                    continue
                if not any(w in low for w in _WORDS):
                    continue
                found.setdefault(name, block[:160])
    return found


def detect(template: str) -> dict:
    """判定模板支持哪种思考控制。返回

        {"kind": "effort"|"switch"|"none",
         "var": "变量名"（none 时为空串）,
         "values": ["low","medium","high"] 或 [True, False] 或 [],
         "evidence": "模板里的片段（给人看/给排查用）",
         "all": ["所有候选变量名"]}
    """
    cands = _candidates(template)
    if not cands:
        return {"kind": "none", "var": "", "values": [], "evidence": "",
                "all": []}
    # effort 类优先（gpt-oss 同时有 reasoning_effort 之类的名字时按这个走）
    effort = sorted(n for n in cands
                    if any(w in n.lower() for w in ("effort", "level", "budget",
                                                    "intensity")))
    switch = sorted(n for n in cands if n not in effort)
    if effort:
        var = effort[0]
        vals = sorted(set(re.findall(
            rf"{re.escape(var)}\s*==\s*[\"'](\w+)[\"']", template or "")))
        return {"kind": "effort", "var": var,
                "values": vals or list(EFFORT_VALUES),
                "evidence": cands[var], "all": sorted(cands)}
    var = switch[0]
    return {"kind": "switch", "var": var, "values": [True, False],
            "evidence": cands[var], "all": sorted(cands)}


def think_kind(template: str) -> str:
    """只要 kind 的便捷入口（UI 用）。"""
    return detect(template)["kind"]


def think_info_for(cfg: dict, model: str) -> dict:
    """模型名 → detect() 的完整结果（按 (路径, 大小, mtime) 缓存）。"""
    try:
        path = KaiwuManager._resolve_gguf(cfg, model)
    except Exception:
        path = ""
    if not path:
        return {"kind": "none", "var": "", "values": [], "evidence": "",
                "all": [], "path": ""}
    try:
        st = Path(path).stat()
        key = (path, st.st_size, st.st_mtime_ns)
    except OSError:
        return {"kind": "none", "var": "", "values": [], "evidence": "",
                "all": [], "path": path}
    with _lock:
        hit = _cache.get(key)
    if hit is None:
        try:
            hit = detect(read_chat_template(path))
        except Exception:
            hit = {"kind": "none", "var": "", "values": [], "evidence": "",
                   "all": []}
        hit["path"] = path
        with _lock:
            _cache[key] = hit
    return hit


def think_kind_for(cfg: dict, model: str) -> str:
    """只要 kind 的便捷入口（/api/models 用）。"""
    return think_info_for(cfg, model)["kind"]


# ============================================================ 命令行工具
def _cli(argv: list[str]) -> int:
    from .config import load_config
    cfg = load_config()
    target = " ".join(argv).strip()
    if target and (target.lower().endswith(".gguf") or Path(target).is_file()):
        info = detect(read_chat_template(target))
        print(f"文件：{target}")
        _print_detail(info)
        return 0
    if target:
        info = think_info_for(cfg, target)
        print(f"模型：{target}")
        print(f"文件：{info.get('path') or '(找不到该 GGUF)'}")
        _print_detail(info)
        return 0
    try:
        models = KaiwuManager().list_models(cfg)
    except Exception as e:
        print("列模型失败：", e)
        return 1
    print("本地模型的思考控制能力（读 GGUF 模板判定，不加载模型）\n")
    print(f"{'模型':28} {'能力':8} {'变量':22} 可选值")
    print("-" * 78)
    for m in models:
        info = think_info_for(cfg, m["name"])
        if info["kind"] == "switch":
            vals = "on / off"
        elif info["kind"] == "effort":
            vals = " / ".join(str(v) for v in info["values"])
        else:
            vals = "—"
        print(f"{m['name'][:27]:28} {info['kind']:8} "
              f"{(info['var'] or '—')[:21]:22} {vals}")
    print("\n提示：能力为 none 的模型，Web UI 会把「思考强度」控件整个藏起来；"
          "\n要单独排查某个模型（看证据片段）：python -m app.model_caps <模型名>")
    return 0


def _print_detail(info: dict) -> None:
    names = {"effort": "三档（low/medium/high）", "switch": "开关（on/off）",
             "none": "没有可调的思考开关"}
    print(f"判定：{info['kind']} —— {names.get(info['kind'], info['kind'])}")
    if info["var"]:
        if info["kind"] == "switch":
            vals = "on / off"
        else:
            vals = " / ".join(str(v) for v in info["values"])
        print(f"变量：{info['var']}    可选值：{vals}")
    if info.get("all") and len(info["all"]) > 1:
        print(f"（其它候选变量：{'、'.join(info['all'])}）")
    if info["evidence"]:
        print(f"证据：…{info['evidence']}…")
    if info["kind"] == "none":
        print("说明：模板里没有任何思考控制变量 —— 要么思考是强制的，要么这个模型"
              "本来就不思考。UI 会把这个控件藏掉。")


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
