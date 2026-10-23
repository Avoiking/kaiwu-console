"""环境信息（当前日期时间）注入 —— 拆掉"模型自以为在 2024 年"的那条回路。

## 为什么要有这个模块

本地模型的权重有**训练截止时间**，它本质上不知道"今天"是几号。实测
qwen3.5-4b-ud 会自称"今天是 2024 年 5 月"，还声称"这是系统提示词里
说的"，然后把搜索词写成「AI圈 2024年 最新新闻」——搜回来自然全是 2024 的旧闻，
旧闻又反过来"证实"它是对的。**幻觉 → 按幻觉搜 → 搜到的东西支持幻觉**，
自我强化之后它改口两次，越说越像真的。

两条待查项已核实，注入来源为零：
1. 全代码库所有注入 system 的文本**一个日期都没有**（只有代码注释里有日期，
   那些从不发给模型）；
2. 模型自带的对话模板实测 dump（GGUF `tokenizer.chat_template`，7816 字符）
   **零日期变量**，也没有 `strftime` / `datetime` / 默认 system 注入，
   只有一句 "answer the question like normal with your current knowledge"。

⇒ 判定为纯幻觉 + 搜索确认回路。治法就是服务端**每次请求**把当前时间喂给它。

## 两条铁律

1. **绝不写死日期**。一律 `time.strftime` 现场取，
   否则明年就变成新的错误信息，比不注入还糟。
2. **不要点名错误年份**去纠正（比如"不要以为现在是 2024 年"）——那等于把
   错误答案又提示一遍。只说正确时间，不说它错在哪。

## 注入点在**最后一条 user 消息的开头**（2026-10-17 从 system 最前面迁来）

原实现把时间块放在 system 消息最前面。它每请求都在变 ⇒ 整个 prompt 的
**第一个 token 就对不上**，llama.cpp 的 prompt cache 永远命中不了 ——
长会话每轮都把整段历史重新预填一遍。真机实测（gemma-4-26B-A4B-IQ3_S，
--cpu-moe @98304，90K 历史）：时间块在 system 最前每轮全量预填 ~135s；
在最后一条 user 开头则前缀稳定、缓存命中 ~2s，差 70 倍。Web UI 的
tok/s 统计曾把预填算进"输出速度"，用户看到的就是"100K 上下文输出
1 tok/s"的假象（解码实际 17~19 tok/s，该统计已一并修正）。

## 为什么只在 Web UI 注入，不进网关

qwen3.5-4b 的模板**只渲染 messages[0] 的 system**（其余 system 直接忽略）。
网关若给第三方客户端（HanaAgent 等）前置一条 system，这条会顶掉客户端自己的
系统提示词 —— 是把别人的 agent 弄坏，而不是修好。所以注入点必须是
"我们自己拼请求"的地方，即 `app/webui.py` 的 `_chat()`。
"""
from __future__ import annotations

import time

# 中文星期（tm_wday: 0=周一）
_WEEK_CN = ("一", "二", "三", "四", "五", "六", "日")


def now_text(ts: float | None = None) -> str:
    """当前本地时间文本：`2026-09-14 07:06（星期日 UTC+0800）`。

    ts 仅供测试注入固定时刻用；生产路径一律传 None（现场取时间）。
    """
    lt = time.localtime(ts)
    stamp = time.strftime("%Y-%m-%d %H:%M", lt)
    week = _WEEK_CN[lt.tm_wday]
    zone = time.strftime("%z", lt)
    zone = f"UTC{zone[:3]}:{zone[3:]}" if len(zone) == 5 else (zone or "")
    return f"{stamp}（星期{week}{(' ' + zone) if zone else ''}）"


def user_prefix(ts: float | None = None) -> str:
    """每次请求现场生成的新鲜度块，注入**最后一条 user 消息的开头**。

    放这里是故意的：system 最前面放变化内容会让 llama.cpp 的 prompt cache
    全程失效（第一个 token 就对不上），长会话每轮全量重填历史。真机实测
    （gemma-4-26B-A4B，--cpu-moe @98304，90K 历史）：旧放法每轮预填
    ~135s，新放法前缀稳定、缓存命中 ~2s，差 70 倍。而 Web UI 的 tok/s
    统计若把预填算进"输出速度"，用户看到的就是"100K 上下文 1 tok/s"——
    解码其实一直是 17~19 tok/s。放最后一条 user 开头（而非 system 末尾）
    是为了稳定前缀最大化：渲染顺序 [system][u1][a1]…[uN]，块在 uN 开头，
    跨轮共同前缀覆盖到 uN-1 的回答；且所有模板都渲染 user 消息。
    """
    return (
        f"【当前时间】{now_text(ts)}\n"
        "这是本机时钟的实时时间，每次请求都会重新给一遍。"
        "你的训练数据有截止时间，并不知道现在是什么时候："
        "凡涉及「最新 / 最近 / 今年 / 现在 / 近期」的问题，一律以上面这个时间为准；"
        "联网搜索时，用它构造关键词（带上当前年份），否则很容易搜回过时的旧闻。"
        "若历史对话里出现过别的日期，以本行时间为准。\n\n"
    )
