"""时间注入位置的规格测试（2026-10-17，prompt cache 修复）。

背景：用户报"100K 上下文输出只有 1 tok/s"。真机排查结论——不是模型慢
（解码实测 17~19 tok/s），是 Web UI 把**预填时间**算进了 tok/s 统计
（completion_tokens ÷ 总耗时），而预填之所以每轮全量发生，是因为时间块
放在 system 最前面、每请求都变，llama.cpp 的 prompt cache 第一个 token
就对不上。实测（gemma-4-26B-A4B，--cpu-moe @98304，90K 历史）：

    时间块在 system 最前（旧）：每轮全量预填 ≈ 135s
    时间块在最后一条 user 开头（新）：前缀稳定、缓存命中 ≈ 2s

本测试锁住的不变式：
  1. system 消息跨请求逐字节稳定（缓存前提），且不含时间块
  2. 时间块只在最后一条 user 消息开头；历史部分原样不动；返回块文本
  3. 两次不同时间的注入，剥掉块后消息列表完全一致
  4. 多模态 list content、无 user 消息兜底、开关关闭 都正确
  5. **跨轮渲染稳定性**（前端把块并进 history 后）：第 N+1 轮的渲染文本
     以第 N 轮整体为前缀 —— prompt cache 能整段复用的充要条件
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.webui as webui
from app import env_info

PASS = FAIL = 0


def ok(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}")


CFG = {"inject_datetime": True}
BLOCK_A = "【当前时间】2026-10-17 09:00（星期六）\n这是本机时钟…\n\n"
BLOCK_B = "【当前时间】2026-10-17 09:07（星期六）\n这是本机时钟…\n\n"

HIST = [
    {"role": "user", "content": "第一轮问题"},
    {"role": "assistant", "content": "第一轮回答"},
    {"role": "user", "content": "第二轮问题"},
]
LONG = "人工智能是研究、开发用于模拟、延伸和扩展人的智能。" * 200


def render(msgs):
    """近似渲染：把所有消息文本按顺序拼接（prompt cache 比的是 token 序列，
    文本一致 ⟺ 前缀一致，测试里用字符串前缀断言足够）。"""
    out = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        out.append(str(c or ""))
    return "".join(out)


print("\n[1] system 消息只含静态部分（跨请求稳定 = prompt cache 前提）")
s1 = webui.compose_system_prompt(CFG, "你是助手", "", "")
s2 = webui.compose_system_prompt(CFG, "你是助手", "", "")
ok(s1 == s2 == "你是助手", f"用户自定义原样输出且稳定（实际 {s1!r}）")
ok("当前时间" not in s1 and "2026" not in s1, "system 里没有时间块")
ok(webui.compose_system_prompt(CFG, "", "", "") == "",
   "无自定义无工具 → 无 system 消息（不硬塞空 system）")

print("\n[2] 时间块注入最后一条 user 开头，历史原样不动，返回块文本")
out, block = webui.inject_freshness(list(HIST), CFG)
ok(out[0]["content"] == "第一轮问题" and out[1]["content"] == "第一轮回答",
   "前面的消息逐字节不动")
ok(out[-1]["content"] == block + "第二轮问题",
   "最后一条 user = 块 + 原问题，且返回的块文本一致")
ok(block.startswith("【当前时间】") and block.endswith("\n\n"), "块格式：时间行开头、空行结尾")
ok(HIST[-1]["content"] == "第二轮问题", "入参没有被就地修改")

print("\n[3] 跨轮不变式：两次不同时间的注入，剥掉块后完全一致")
calls = {"n": 0}
_orig = webui.env_info.user_prefix


def _fake_prefix(ts=None):
    calls["n"] += 1
    return BLOCK_A if calls["n"] == 1 else BLOCK_B


webui.env_info.user_prefix = _fake_prefix
try:
    a, ba = webui.inject_freshness(list(HIST), CFG)
    b, bb = webui.inject_freshness(list(HIST), CFG)
finally:
    webui.env_info.user_prefix = _orig
ok(a[-1]["content"][len(BLOCK_A):] == b[-1]["content"][len(BLOCK_B):] == "第二轮问题",
   "剥掉各自的时间块后，最后一条 user 完全一致（前缀稳定）")
ok(a[0] == b[0] and a[1] == b[1], "历史消息两次注入零差异")
ok(ba != bb, "块本身确实每次不同（时间保鲜）")

print("\n[4] 多模态 list content：块作为 text part 插在最前")
mm = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:x"}},
                                    {"type": "text", "text": "图里有啥"}]}]
out_mm, _ = webui.inject_freshness(mm, CFG)
parts = out_mm[-1]["content"]
ok(isinstance(parts, list) and parts[0]["type"] == "text"
   and "【当前时间】" in parts[0]["text"], "块是第一个 text part")
ok(parts[1]["type"] == "image_url" and parts[2]["text"] == "图里有啥",
   "图片 part 与原 text part 顺序内容不动")

print("\n[5] 边界：无 user 消息 / 开关关闭")
sys_only = [{"role": "system", "content": "静态系统提示"}]
out_sys, blk_sys = webui.inject_freshness(sys_only, CFG)
ok(out_sys[0]["role"] == "system" and "【当前时间】" in out_sys[0]["content"]
   and blk_sys, "没有 user 消息 → 退回拼一条带时间块的 system（不丢时间）")
ok(len(out_sys) == 2 and out_sys[1]["content"] == "静态系统提示", "原 system 消息保留")
off = {"inject_datetime": False}
ok(webui.inject_freshness(list(HIST), off) == (list(HIST), ""),
   "开关关闭 → 原样返回 + 空块")

print("\n[6] 跨轮渲染稳定性（前端把块并进 history 后）—— 缓存命中的充要条件")
m1, b1 = webui.inject_freshness([{"role": "user", "content": LONG}], CFG)
# 前端收到 kaiwu_fresh 后把块并进 history：u1 带块存起来
hist2 = [dict(m1[0]), {"role": "assistant", "content": "好的，已总结。"}]
m2, b2 = webui.inject_freshness(hist2 + [{"role": "user", "content": "再补充一句。"}], CFG)
r1, r2 = render(m1), render(m2)
ok(r2.startswith(r1),
   "第二轮渲染以第一轮整体为前缀（38K 历史整段复用，只填新增部分）")
ok(render(m2).endswith(b2 + "再补充一句。"), "新问题的块在最末（新鲜度落在最后一轮）")
# 反向对照：前端**不**并块（旧 bug 形态）→ 第 0 个 token 就分叉
hist_noblock = [{"role": "user", "content": LONG},
                {"role": "assistant", "content": "好的，已总结。"}]
m2b, _ = webui.inject_freshness(hist_noblock + [{"role": "user", "content": "再补充一句。"}], CFG)
ok(not render(m2b).startswith(r1[:len(LONG)]),
   "对照：不并块则历史消息与上轮发出版本分叉（缓存 miss，每轮全量重填）")

print("\n[7] 时间块内容本身：现场取、不写死、指令完整")
blk = env_info.user_prefix(1_700_000_000.0)     # 固定时刻，确定性断言
ok(blk.startswith("【当前时间】"), "块以时间行开头")
ok("2023" in blk, f"固定时刻渲染正确（实际 {blk[:30]!r}）")
ok("以本行时间为准" in blk, "保留'历史里出现别的日期以本行为准'的指令")

print(f"\n===== {PASS} 通过 / {FAIL} 失败 =====")
if FAIL:
    print("注：任一 FAIL 都意味着 prompt cache 可能再次被打碎 —— 长会话每轮")
    print("    全量重填 100K（~135s），tok/s 统计又会显示成 1 tok/s 的假象。")
sys.exit(1 if FAIL else 0)
