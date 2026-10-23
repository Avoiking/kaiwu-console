"""Ollama 协议网关 + OpenAI 直通，让 kaiwu 对外提供两种 API。

让 HANA AGENT / Chatbox / Talkio / ollama CLI 等客户端接入 kaiwu：

Ollama 协议（自动按请求里的模型名切换加载）：
- GET  /  /api/version  /api/tags  /api/ps   POST /api/show  /api/chat  /api/generate
- 流式 NDJSON / 非流式 JSON；images 字段（视觉模型可发图）、tools 透传

OpenAI 直通（/v1/*，原样转发给 llama-server）：
- GET /v1/models：**网关自己答，返回本地全部模型**（不走 llama-server）。
  llama-server 只报当前加载的那一个，客户端「读取模型」就得逐个加载才能读全；
  这里一次给全，方便 HanaAgent 之类一次性把模型列表加好。
- POST /v1/chat/completions /v1/completions：**也按请求里的 model 名自动切换**
  （config.gateway_v1_model_switch，默认开）——让只讲 OpenAI 协议的客户端
  （HanaAgent 等）也能自己换模型。名字解析不到本地 gguf 时不报错、原样放行
- GET /v1/api/tags：Talkio 的 Ollama 预设把发现路径拼成 /v1/api/tags
  （对官方 Ollama 是 404，其预设自身的 bug），这里给个别名兜住
- 其余 /v1/* 路径原样透传，由当前加载的模型答（换模型回控制台 use）

思考链归位（config.normalize_think，默认开）：
- 有些本地模型的模板没声明思考标记（如 LFM2.5-8B-A1B，llama-server 报 thinking=0），
  于是 <think>…</think> 会整段落进**正文**，客户端只能当普通文本看、还白占上下文。
  网关在返回时把开头的 <think> 挪进 reasoning_content；**/v1 与 Ollama 两条路都生效**，
  流式为增量解析（标签跨块也不会漏）。对本来就被正确解析的模型（qwen3.5-4b-ud /
  gpt-oss-20b）是空操作；设 false 走旧的整块透传路径

工具调用：**一律原样透传**给上游，网关不做任何归位。
- 2026-09-13：原先有一套针对 LFM2.5 的"文本工具调用归位"（它把调用写在正文里，
  上游解析器拆不动会直接 500），随 LFM2.5 模型退役一并删除（备份见
  `app/ollama_gateway.py.bak_20260913_lfmretire`）。
  现在只剩原生路径：模型必须能被 llama-server 的模板解析器正确解析。

端口策略：优先 config.gateway_port（默认 11434，客户端零配置），
被占用（真 Ollama 在跑）则退到 gateway_fallback_port（默认 11437）。
注意：同一时刻只有一个模型在显存里，切换会切断进行中的流式回复。
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from .config import load_config
from .kaiwu_manager import KaiwuManager

GATEWAY_VERSION = "0.5.7-kaiwu"

_FAMILY_PATTERNS = [
    (r"gpt-?oss", "gptoss"),
    (r"qwen", "qwen"),
    (r"josie", "qwen"),
    (r"lfm", "lfm"),
    (r"smol", "smollm"),
]

# ---------------------------------------------------------------- 工具调用 id
# 陷阱：早先用轮内序号 f"call_{i}"，本轮第一个调用永远是 call_0。
# 客户端（如 HanaAgent）拿 toolCallId 当结果缓存的 key，跨轮撞车后
# 第二次调用起会命中第一次的旧结果 —— 表现为"所有工具都返回同一份内容"，
# 任何模型在这种污染下都会显得很笨。故 id 必须全局唯一。
_TOOL_ID_SEQ = 0
_TOOL_ID_LOCK = threading.Lock()


def _new_tool_call_id(i: int = 0) -> str:
    """生成全局唯一的工具调用 id（跨轮、跨会话都不会重复）。"""
    global _TOOL_ID_SEQ
    with _TOOL_ID_LOCK:
        _TOOL_ID_SEQ += 1
        seq = _TOOL_ID_SEQ
    raw = f"{time.time_ns()}-{seq}-{i}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
    return f"call_{digest}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _family_of(name: str) -> str:
    n = name.lower()
    for pat, fam in _FAMILY_PATTERNS:
        if re.search(pat, n):
            return fam
    return "llama"


def _param_of(name: str, size: int) -> str:
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name.lower())
    if m:
        return f"{m.group(1)}B"
    approx = size / (1024 ** 3) / 0.6  # Q4 量化粗略 0.6 byte/param
    for std in (0.6, 1, 1.5, 2, 3, 4, 7, 8, 9, 12, 14, 20, 32, 70):
        if abs(approx - std) / std < 0.3:
            return f"{std}B"
    return f"{approx:.0f}B"


def _quant_of(name: str) -> str:
    m = re.search(r"(q\d[_a-z0-9]*|f16|bf16|fp8|iq[_a-z0-9]*)", name.lower())
    return m.group(1).upper() if m else "Q4_0"


def _mime_of_b64(b64: str) -> str:
    if b64.startswith("/9j/"):
        return "image/jpeg"
    if b64.startswith("iVBOR"):
        return "image/png"
    if b64.startswith("R0lGOD"):
        return "image/gif"
    if b64.startswith("UklGR"):
        return "image/webp"
    return "image/png"


def _map_options(options: dict) -> dict:
    """Ollama options → llama-server 认识的采样参数；num_ctx/keep_alive 忽略。"""
    out = {}
    for k in ("temperature", "top_p", "top_k", "repeat_penalty", "min_p"):
        if k in options:
            out[k] = options[k]
    if "num_predict" in options:
        out["max_tokens"] = options["num_predict"]
    if "stop" in options:
        out["stop"] = options["stop"]
    if options.get("seed") not in (None, -1, 0, False):
        out["seed"] = options["seed"]
    return out


def _ensure_max_tokens(req: dict) -> None:
    """客户端没给生成长度时兜底一个宽松默认值。

    思考型模型（josie-thinking / qwen 思考链 / gpt-oss）会把生成 token 花在
    思考上，客户端没显式设置或设太小时，思考一结束就被截断、正文为空。
    这里只兜缺省（setdefault），不覆盖客户端显式值；上游 llama-server 会
    自动把 n_predict 钳制到剩余上下文内，不会越界。
    """
    cfg = load_config()
    default = int(cfg.get("max_tokens_default") or 8192)
    if not req.get("max_tokens"):
        req["max_tokens"] = default


def _conv_msg(m: dict) -> dict:
    """Ollama 消息 → OpenAI 消息（images 转多模态数组，tool_calls 转标准形状）。"""
    role = m.get("role") or "user"
    content = m.get("content") or ""
    images = m.get("images") or []
    out: dict = {"role": role, "content": content}
    if images:
        parts = [{"type": "text", "text": content}] if content else []
        for b64 in images:
            if isinstance(b64, str) and b64:
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:{_mime_of_b64(b64)};base64,{b64}"}})
        out["content"] = parts
    if m.get("tool_calls"):
        tcs = []
        for i, tc in enumerate(m["tool_calls"]):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            tcs.append({"id": tc.get("id") or _new_tool_call_id(i), "type": "function",
                        "function": {"name": fn.get("name") or "",
                                     "arguments": args or "{}"}})
        out["tool_calls"] = tcs
        if not out["content"]:
            out["content"] = ""
    if role == "tool":
        out["tool_call_id"] = m.get("tool_call_id") or ""
    return out


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def gw(self) -> "OllamaGateway":
        return self.server.gateway  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):  # 安静：仅网关内部日志可见
        pass

    # ------------------------------------------------------------ 路由
    def _check_auth(self) -> bool:
        """API key 门禁：仅对非本机（非回环）请求生效。

        - config.gateway_api_key 为空 → 不启用，行为同旧版
        - 回环地址（127.0.0.1 / ::1）始终放行：ollama CLI / HANA / Talkio 本机零配置
        - 其他来源必须带 Authorization: Bearer <key> 或 X-Api-Key: <key>
        """
        key = str(load_config().get("gateway_api_key") or "").strip()
        if not key:
            return True
        peer = self.client_address[0] if self.client_address else ""
        if peer in ("127.0.0.1", "::1", "localhost"):
            return True
        auth = self.headers.get("Authorization") or ""
        if auth.startswith("Bearer ") and auth[7:].strip() == key:
            return True
        if (self.headers.get("X-Api-Key") or "").strip() == key:
            return True
        self._json(401, {"error": "unauthorized: missing or invalid API key "
                                  "(Authorization: Bearer <key> or X-Api-Key)"})
        return False

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self):
        # ollama CLI 0.32.x 每次命令先发 HEAD / 探活，缺这个方法会 501，
        # CLI 直接报 "something went wrong"（HEAD 无 body，探活放行）
        self.send_response(200)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if not self._check_auth():
            self.gw.flog(f"GET {self.path.split('?', 1)[0]} <- {self.client_address[0]} -> 401")
            return
        self.gw.flog(f"GET {self.path.split('?', 1)[0]} <- {self.client_address[0]}")
        path = self.path.split("?", 1)[0]
        if path in ("/", "/api"):
            self._text(200, "Ollama is running")
        elif path == "/api/version":
            self._json(200, {"version": GATEWAY_VERSION})
        elif path == "/api/tags":
            self._json(200, self.gw.tags_payload())
        elif path == "/api/ps":
            self._json(200, self.gw.ps_payload())
        elif path == "/v1/api/tags":
            # Talkio 的 Ollama 预设把 /api/tags 拼到以 /v1 结尾的 baseUrl 后
            # （对官方 Ollama 也是 404，其预设自身的 bug），这里给个别名兜住
            self._json(200, self.gw.tags_payload())
        elif path == "/v1/models":
            # 一次读全：llama-server 的 /v1/models 只回当前加载的那 1 个，
            # 客户端加模型就得「加载 A→读取→加载 B→读取」来回切；这里直接
            # 返回本地全部模型名。仅列表用途，不发请求、不触发加载
            self._json(200, self.gw.openai_models_payload())
        elif path.startswith("/v1/"):
            self._proxy("GET", path)
        else:
            self._json(404, {"error": f"unknown endpoint {path}"})

    def do_POST(self):
        if not self._check_auth():
            self.gw.flog(f"POST {self.path.split('?', 1)[0]} <- {self.client_address[0]} -> 401")
            return
        self.gw.flog(f"POST {self.path.split('?', 1)[0]} <- {self.client_address[0]}")
        path = self.path.split("?", 1)[0]
        if path == "/api/show":
            self._show()
        elif path == "/api/chat":
            self._chat()
        elif path == "/api/generate":
            self._generate()
        elif path in ("/api/embed", "/api/embeddings"):
            self._json(501, {"error": "embeddings are not supported by this gateway"})
        elif path.startswith("/v1/"):
            body = None
            if path in ("/v1/chat/completions", "/v1/completions"):
                # 先读完请求体（_proxy 需要复用同一份），再按 model 名决定要不要切换
                body = self._raw_body()
                if not self._maybe_switch_openai(body):
                    return
            self._proxy("POST", path, body=body)
        else:
            self._json(404, {"error": f"unknown endpoint {path}"})

    # ------------------------------------------------------------ 端点
    def _show(self):
        body = self._read_json()
        # ollama CLI 0.32.x 的 /api/show 把模型名放 name 字段（旧版用 model）
        want = body.get("model") or body.get("name") or ""
        entry = self.gw.resolve_model(want)
        if entry is None:
            self._json(404, {"error": f'model "{want}" not found'})
            return
        det = self.gw._details(entry)
        self._json(200, {
            "license": "", "modelfile": "", "parameters": "", "template": "",
            "details": det,
            "model_info": {"general.architecture": det["family"],
                           "general.parameter_count": 0, "general.file_type": 1},
        })

    def _chat(self):
        body = self._read_json()
        if self._maybe_unload(body, chat_mode=True):
            return
        entry = self._prepare(body)
        if entry is None:
            return
        want_stream = bool(body.get("stream", True))
        messages = [_conv_msg(m) for m in (body.get("messages") or []) if isinstance(m, dict)]
        req = {"model": entry["name"], "messages": messages, "stream": want_stream}
        req.update(_map_options(body.get("options") or {}))
        if body.get("format") == "json":
            req["response_format"] = {"type": "json_object"}
        if body.get("tools"):
            req["tools"] = body["tools"]
        self._apply_think(body, req)
        _ensure_max_tokens(req)
        self._run(req, entry["name"], want_stream, chat_mode=True)

    def _generate(self):
        body = self._read_json()
        if self._maybe_unload(body, chat_mode=False):
            return
        entry = self._prepare(body)
        if entry is None:
            return
        want_stream = bool(body.get("stream", True))
        messages = []
        if body.get("system"):
            messages.append({"role": "system", "content": str(body["system"])})
        messages.append(_conv_msg({"role": "user", "content": body.get("prompt") or "",
                                   "images": body.get("images") or []}))
        req = {"model": entry["name"], "messages": messages, "stream": want_stream}
        req.update(_map_options(body.get("options") or {}))
        if body.get("format") == "json":
            req["response_format"] = {"type": "json_object"}
        self._apply_think(body, req)
        _ensure_max_tokens(req)
        self._run(req, entry["name"], want_stream, chat_mode=False)

    def _apply_think(self, body: dict, req: dict) -> None:
        """Ollama 的 think:false → llama-server 模板参数（Qwen3 家族用
        enable_thinking 控制思考链）；think:true/缺省不动，保持模型默认。"""
        if body.get("think") is False:
            req["chat_template_kwargs"] = {"enable_thinking": False}

    def _maybe_unload(self, body: dict, chat_mode: bool) -> bool:
        """Ollama 卸载语义：keep_alive=0 → 停止推理服务释放显存（不生成）。

        官方 Ollama 收到 keep_alive=0 会卸载模型；原实现忽略该字段，客户端
        （如 OCV 出图前的显存腾挪）不仅卸不掉，若模型未加载还会被
        ensure_loaded 反向拉起。这里同步 stop（最长约 20s），再回空的
        done 响应。带正文的 keep_alive=0（生成后卸载）不支持，返回空。
        """
        ka = body.get("keep_alive")
        if isinstance(ka, bool) or (ka != 0 and str(ka) != "0"):
            return False
        name = str(body.get("model") or "")
        self.gw.manager.stop()
        out: dict = {
            "model": name, "created_at": _now(), "done": True, "done_reason": "unload",
            "total_duration": 0, "prompt_eval_count": 0, "eval_count": 0,
        }
        if chat_mode:
            out["message"] = {"role": "assistant", "content": ""}
        else:
            out["response"] = ""
        self._json(200, out)
        return True

    def _prepare(self, body: dict) -> dict | None:
        """解析模型名 → 确保已加载。失败时已把错误写给客户端，返回 None。"""
        want = str(body.get("model") or "")
        entry = self.gw.resolve_model(want)
        if entry is None:
            self._json(404, {"error": f'model "{want}" not found'})
            return None
        if not self.gw.ensure_loaded(entry["name"]):
            err = self.gw.manager.last_error or "unknown error"
            self._json(503, {"error": f'failed to load model "{entry["name"]}": {err}'})
            return None
        return entry

    # ------------------------------------------------------------ 上游调用与翻译
    def _run(self, req: dict, model: str, want_stream: bool, chat_mode: bool) -> None:
        cfg = load_config()
        base = self.gw.manager.base_url(cfg)
        t0 = time.time()
        if want_stream:
            req["stream_options"] = {"include_usage": True}  # 末块带 usage，流式统计才准
        try:
            upstream = requests.post(
                f"{base}/v1/chat/completions", json=req,
                stream=True, timeout=(10, 3600),
                headers={"Accept": "text/event-stream"},
            )
        except Exception as e:
            self._json(502, {"error": f"llama-server 请求失败：{e}"})
            return
        if upstream.status_code != 200:
            detail = upstream.text[:400]
            upstream.close()
            self._json(502, {"error": f"llama-server HTTP {upstream.status_code}：{detail}"})
            return
        upstream.encoding = "utf-8"  # SSE 头无 charset，防中文乱码

        if not want_stream:
            self._respond_once(upstream, model, chat_mode, t0)
            return
        self._respond_stream(upstream, model, chat_mode, t0)

    def _respond_once(self, upstream, model: str, chat_mode: bool, t0: float) -> None:
        try:
            data = upstream.json()
        except Exception as e:
            self._json(502, {"error": f"llama-server 响应解析失败：{e}"})
            return
        finally:
            upstream.close()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        think_raw, content_clean = _split_think_full(msg.get("content") or "")
        # done_reason 透传截断信号：OCV 靠 length 判断输出被 num_predict 截断
        out: dict = {
            "model": model, "created_at": _now(), "done": True,
            "done_reason": "length" if choice.get("finish_reason") == "length" else "stop",
            "total_duration": int((time.time() - t0) * 1e9),
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }
        if chat_mode:
            thinking = (msg.get("reasoning_content") or "") + think_raw
            om: dict = {"role": "assistant", "content": content_clean}
            if thinking:
                om["thinking"] = thinking
            if msg.get("tool_calls"):
                om["tool_calls"] = [
                    {"function": {"name": (tc.get("function") or {}).get("name") or "",
                                  "arguments": _args_to_dict((tc.get("function") or {}).get("arguments"))}}
                    for tc in msg["tool_calls"]]
            out["message"] = om
        else:
            out["response"] = content_clean
        self._json(200, out)

    def _respond_stream(self, upstream, model: str, chat_mode: bool, t0: float) -> None:
        self._ndjson_start()
        stats: dict = {}
        finish_reason: str = ""
        tool_calls: list[dict] = []
        splitter = _ThinkSplitter()
        normalize_think = bool(load_config().get("normalize_think", True))
        try:
            for raw in upstream.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                if obj.get("error"):
                    self._ndjson_line({"model": model, "created_at": _now(),
                                       "error": str(obj["error"]), "done": True})
                    return
                choices = obj.get("choices") or []
                if not choices:
                    if obj.get("usage"):
                        stats = obj["usage"]
                    continue
                if choices[0].get("finish_reason"):
                    finish_reason = str(choices[0]["finish_reason"])
                delta = choices[0].get("delta") or {}
                rc = delta.get("reasoning_content")
                if rc and chat_mode:
                    self._ndjson_line({"model": model, "created_at": _now(),
                                       "message": {"role": "assistant", "thinking": rc},
                                       "done": False})
                c = delta.get("content")
                if c:
                    t_extra, c_extra = splitter.feed(c)
                    if t_extra and chat_mode:
                        self._ndjson_line({"model": model, "created_at": _now(),
                                           "message": {"role": "assistant", "thinking": t_extra},
                                           "done": False})
                    if c_extra:
                        piece = ({"message": {"role": "assistant", "content": c_extra}}
                                 if chat_mode else {"response": c_extra})
                        self._ndjson_line({"model": model, "created_at": _now(),
                                           **piece, "done": False})
                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        idx = int(tc.get("index") or 0)
                        while len(tool_calls) <= idx:
                            tool_calls.append({"function": {"name": "", "arguments": ""}})
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            tool_calls[idx]["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            tool_calls[idx]["function"]["arguments"] += fn["arguments"]
                if obj.get("usage"):
                    stats = obj["usage"]
            tail = splitter.flush()
            if tail and chat_mode:
                self._ndjson_line({"model": model, "created_at": _now(),
                                   "message": {"role": "assistant", "thinking": tail},
                                   "done": False})
            final: dict = {
                "model": model, "created_at": _now(), "done": True,
                "done_reason": "length" if finish_reason == "length" else "stop",
                "total_duration": int((time.time() - t0) * 1e9),
                "prompt_eval_count": stats.get("prompt_tokens", 0),
                "eval_count": stats.get("completion_tokens", 0),
            }
            if chat_mode:
                om: dict = {"role": "assistant", "content": ""}
                if tool_calls:
                    om["tool_calls"] = [{"function": {
                        "name": tc["function"]["name"],
                        "arguments": _args_to_dict(tc["function"]["arguments"])}}
                        for tc in tool_calls]
                final["message"] = om
            else:
                final["response"] = ""
            self._ndjson_line(final)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # 客户端断开
        except Exception as e:
            try:
                self._ndjson_line({"model": model, "created_at": _now(),
                                   "error": str(e), "done": True})
            except Exception:
                pass
        finally:
            upstream.close()

    # ------------------------------------------------------------ OpenAI 直通（/v1/* 原样转发）
    def _proxy(self, method: str, path: str, body: bytes | None = None) -> None:
        """把 /v1/* 的 OpenAI 协议请求原样转发给 llama-server。

        body=None 时自己读请求体；调用方已读过（自动切换前要解析 model 字段）
        就传进来复用，避免 body 被读两遍变成空。

        切换行为：/v1/chat/completions 与 /v1/completions 会先按请求里的
        model 名自动切换（config.gateway_v1_model_switch，默认开）；
        其它 /v1/* 路径与解析不到的名字一律原样透传，由当前加载的模型答。
        """
        cfg = load_config()
        base = self.gw.manager.base_url(cfg)
        if body is None:
            body = self._raw_body() if method == "POST" else b""
        try:
            upstream = requests.request(
                method, f"{base}{path}", data=body or None,
                headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
                stream=True, timeout=(10, 3600))
        except Exception as e:
            self._json(502, {"error": {"message": f"llama-server 请求失败：{e}",
                                       "type": "upstream_error"}})
            return
        if upstream.status_code >= 400:
            detail = upstream.text[:600]
            code = upstream.status_code
            upstream.close()
            self._json(code, {"error": {"message": f"llama-server HTTP {code}",
                                        "type": "upstream_error", "detail": detail}})
            return
        ctype = upstream.headers.get("Content-Type", "application/json")
        if "event-stream" in ctype:
            self._proxy_stream(upstream, ctype)
            return
        try:
            data = upstream.content
        except Exception as e:
            upstream.close()
            self._json(502, {"error": {"message": f"读取响应失败：{e}",
                                       "type": "upstream_error"}})
            return
        upstream.close()
        cfg2 = load_config()
        if "json" in ctype:
            if cfg2.get("normalize_think", True):
                data = _normalize_openai_payload(data)
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _proxy_stream(self, upstream, ctype: str) -> None:
        """SSE 透传，带思考链归位：
        正文开头内联的 <think>…</think> 挪到 reasoning_content，
        流式为增量解析（标签跨块也不会漏）。

        normalize_think 关掉时走"整块原样拷贝"路径（零解析开销）。"""
        cfg = load_config()
        normalize_think = bool(cfg.get("normalize_think", True))
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def emit(obj: dict) -> None:
            self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False)
                              + "\n\n").encode("utf-8"))
            self.wfile.flush()

        try:
            if not normalize_think:
                for chunk in upstream.iter_content(chunk_size=8192):
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                return
            upstream.encoding = "utf-8"   # SSE 头无 charset，防中文乱码
            splitter = _ThinkSplitter()
            finish_reason = "stop"
            for raw in upstream.iter_lines(decode_unicode=True):
                if raw is None or not raw.strip():
                    continue
                if not raw.startswith("data:"):
                    continue
                payload = raw[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                for ch in (obj.get("choices") or []):
                    if not isinstance(ch, dict):
                        continue
                    if ch.get("finish_reason"):
                        finish_reason = str(ch["finish_reason"])
                    delta = ch.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    c2 = delta.get("content")
                    if normalize_think and isinstance(c2, str) and c2:
                        if splitter.mode != "content":
                            think_part, content_part2 = splitter.feed(c2)
                            if think_part:
                                delta["reasoning_content"] = (
                                    delta.get("reasoning_content") or "") + think_part
                            if content_part2:
                                delta["content"] = content_part2
                            else:
                                delta.pop("content", None)
                    if not delta:
                        continue
                emit(obj)

            # 思考链尾巴
            if normalize_think:
                tail_think, tail_content = splitter.flush_split()
                if tail_think or tail_content:
                    d: dict = {}
                    if tail_content:
                        d["content"] = tail_content
                    if tail_think:
                        d["reasoning_content"] = tail_think
                    emit({"choices": [{"index": 0, "delta": d, "finish_reason": None}]})

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # 客户端断开
        finally:
            upstream.close()

    # ------------------------------------------------------------ HTTP 小工具
    def _read_json(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        raw = self.rfile.read(n) if n else b""
        try:
            obj = json.loads(raw.decode("utf-8") or "{}")
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _raw_body(self) -> bytes:
        """把原始请求体读出来（/v1/* 透传前需要先看 model 字段，读完还得原样发）。"""
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        return self.rfile.read(n) if n else b""

    def _maybe_switch_openai(self, body: bytes) -> bool:
        """OpenAI 路径（/v1/chat/completions、/v1/completions）的自动切换。

        与 Ollama 路径同样语义——请求里写哪个模型就确保哪个在跑，切完再转发，
        让 HanaAgent 这类只讲 OpenAI 协议的客户端也能自己换模型。

        返回 False 表示已把错误写给客户端，调用方直接返回（不再转发）。
        名字解析不到本地 gguf 时**不报错**，记一行日志后原样放行：
        llama-server 本来就忽略 model 名，客户端随便写名字不该被拦下来。
        """
        cfg = load_config()
        if not cfg.get("gateway_v1_model_switch", True) or not body:
            return True
        try:
            payload = json.loads(body.decode("utf-8", "replace"))
        except Exception:
            return True
        if not isinstance(payload, dict):
            return True
        want = str(payload.get("model") or "").strip()
        if not want:
            return True
        entry = self.gw.resolve_model(want)
        if entry is None:
            self.gw.flog(f"/v1 模型名无法解析，按当前模型处理：{want}")
            return True
        if not self.gw.ensure_loaded(entry["name"]):
            err = self.gw.manager.last_error or "unknown error"
            self._json(503, {"error": {
                "message": f'failed to load model "{entry["name"]}": {err}',
                "type": "upstream_error"}})
            return False
        return True

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, code: int, s: str) -> None:
        body = s.encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _ndjson_start(self) -> None:
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _ndjson_line(self, obj: dict) -> None:
        self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        self.wfile.flush()


def _args_to_dict(args) -> dict:
    if isinstance(args, dict):
        return args
    try:
        v = json.loads(args or "{}")
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _split_think_full(text: str) -> tuple[str, str]:
    """整段文本版：开头 <think>...</think> 拆成 (thinking, content)；未闭合全算 thinking。"""
    if not text:
        return "", ""
    stripped = text.lstrip()
    if not stripped.startswith("<think>"):
        return "", text
    body = stripped[len("<think>"):]
    close = body.find("</think>")
    if close == -1:
        return body, ""
    return body[:close], body[close + len("</think>"):].lstrip("\n")


def _normalize_openai_payload(data: bytes) -> bytes:
    """非流式：把 choices[].message.content 开头的 <think>…</think> 挪进
    reasoning_content。不是 JSON / 没命中就原样返回（零副作用）。"""
    try:
        obj = json.loads(data.decode("utf-8"))
    except Exception:
        return data
    if not isinstance(obj, dict):
        return data
    changed = False
    for ch in (obj.get("choices") or []):
        msg = ch.get("message") if isinstance(ch, dict) else None
        if not isinstance(msg, dict):
            continue
        c = msg.get("content")
        if not isinstance(c, str) or "<think>" not in c:
            continue
        thinking, content = _split_think_full(c)
        if not thinking:
            continue
        msg["reasoning_content"] = (msg.get("reasoning_content") or "") + thinking
        msg["content"] = content
        changed = True
    return json.dumps(obj, ensure_ascii=False).encode("utf-8") if changed else data


def _partial_tag_keep(text: str, tag: str) -> int:
    """text 结尾有 k 个字符正好是 tag 的前缀（= 可能是被切断的标签），
    返回需要留住的长度 k；没有则 0。流式解析防"标签跨块"用。"""
    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-k:]):
            return k
    return 0


class _ThinkSplitter:
    """流式版：把 content 流开头的 <think>...</think> 拆成 thinking，正文原样输出。

    llama-server 对 LFM 等模型不解析 <think>（reasoning auto 检测不到模板支持，
    --reasoning-format deepseek 实测也无效），只能在网关侧剥。
    """
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.mode = "unknown"   # unknown / think / content
        self.buf = ""

    def feed(self, delta: str) -> tuple[str, str]:
        """返回 (thinking_delta, content_delta)。"""
        if not delta:
            return "", ""
        if self.mode == "content":
            return "", delta
        self.buf += delta
        if self.mode == "unknown":
            head = self.buf.lstrip()
            if head.startswith(self.OPEN):
                self.mode, self.buf = "think", head[len(self.OPEN):]
            elif len(head) < len(self.OPEN) and self.OPEN.startswith(head):
                return "", ""  # 还无法判定是否 <think> 开头，继续攒
            else:
                # 判定为普通正文：吐出去的同时**必须清空缓冲**，否则这块文本会
                # 在 flush 时再被吐一遍（原实现漏了清空，首块会重复一次）
                out, self.buf = self.buf, ""
                self.mode = "content"
                return "", out
        if self.mode == "think":
            idx = self.buf.find(self.CLOSE)
            if idx == -1:
                # 没找到闭合标签：可以吐，但**必须留住可能是 </think> 前缀的尾巴**。
                # 否则 "</thi" 会被当成思考吐出去，下一块 "nk>正文" 就再也认不出来
                # —— 实测踩过（单测 chunks 切碎场景）。
                keep = _partial_tag_keep(self.buf, self.CLOSE)
                if keep:
                    out, self.buf = self.buf[:-keep], self.buf[-keep:]
                else:
                    out, self.buf = self.buf, ""
                return out, ""
            think_part, rest = self.buf[:idx], self.buf[idx + len(self.CLOSE):]
            self.mode, self.buf = "content", ""
            return think_part, rest.lstrip("\n")
        return "", self.buf

    def flush(self) -> str:
        """流结束：未闭合的 think 残片按 thinking 返回。"""
        out, self.buf = self.buf, ""
        return out

    def flush_split(self) -> tuple[str, str]:
        """流结束：返回 (thinking, content)。

        与 flush() 的区别：**还没判定出是不是 <think> 开头的一律算正文**——
        那可能只是普通文本的开头（比如首块只到了 "<" 或 "<thi"），
        按 thinking 吐出去就错了。"""
        out, self.buf = self.buf, ""
        if not out:
            return "", ""
        return (out, "") if self.mode == "think" else ("", out)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class OllamaGateway:
    MAX_LOG_BYTES = 5 * 1024 * 1024  # 单文件 5MB，超过轮转为 .1

    def __init__(self, manager: KaiwuManager, log=None):
        self.manager = manager
        self._log = log or (lambda s: None)
        self.port: int | None = None
        self._httpd: _Server | None = None
        self._lock = threading.Lock()
        self._flock = threading.Lock()

    # ------------------------------------------------------------ 日志落盘
    def flog(self, line: str) -> None:
        """请求/事件追加到 data/gateway.log（5MB 轮转），排查历史问题用。"""
        try:
            from .config import config_path
            path = config_path().parent / "gateway.log"
            with self._flock:
                if path.exists() and path.stat().st_size > self.MAX_LOG_BYTES:
                    path.replace(path.with_suffix(".log.1"))
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(path, "a", encoding="utf-8") as f:
                    f.write(f"[{stamp}] {line}\n")
        except Exception:
            pass  # 日志失败绝不影响网关主流程

    # ------------------------------------------------------------ 生命周期
    def start(self) -> int:
        with self._lock:
            if self._httpd is not None:
                return self.port  # type: ignore[return-value]
            cfg = load_config()
            ports: list[int] = []
            for key in ("gateway_port", "gateway_fallback_port"):
                p = int(cfg.get(key) or 0)
                if p and p not in ports:
                    ports.append(p)
            if not ports:
                ports = [11434, 11437]
            last_err: Exception | None = None
            # gateway_host：默认 127.0.0.1 仅本机；设 0.0.0.0 对局域网开放
            # （开放时建议配 gateway_api_key，非本机请求必须带 key）
            host = str(cfg.get("gateway_host") or "127.0.0.1").strip() or "127.0.0.1"
            for port in ports:
                try:
                    httpd = _Server((host, port), _Handler)
                except OSError as e:
                    last_err = e
                    self._log(f"网关端口 {port} 被占用（{e.strerror or e}），尝试备用端口…")
                    continue
                httpd.gateway = self  # type: ignore[attr-defined]
                self._httpd = httpd
                self.port = port
                threading.Thread(target=httpd.serve_forever,
                                 kwargs={"poll_interval": 0.5}, daemon=True).start()
                self._log(f"Ollama 网关已启动：http://{host}:{port}"
                          + ("（局域网开放，需 API key）" if host == "0.0.0.0" else ""))
                self.flog(f"网关启动 {host}:{port}")
                return port
            raise RuntimeError(f"网关端口全部被占用：{last_err}")

    def stop(self) -> None:
        with self._lock:
            if self._httpd is not None:
                self._httpd.shutdown()
                self._httpd.server_close()
                self._httpd = None
                self.port = None
                self.flog("网关停止")

    @property
    def running(self) -> bool:
        return self._httpd is not None

    # ------------------------------------------------------------ 模型解析与清单
    def resolve_model(self, name: str) -> dict | None:
        """客户端模型名（可带 :tag）→ 本地模型条目。"""
        name = (name or "").strip()
        if not name:
            return None
        base = name.split(":", 1)[0].strip()
        models = self.manager.list_models()
        by_lower = {m["name"].lower(): m for m in models}
        for cand in (name, base):
            hit = by_lower.get(cand.lower())
            if hit:
                return hit
        for m in models:  # 宽松包含匹配
            if base and (base.lower() in m["name"].lower() or m["name"].lower() in base.lower()):
                return m
        return None

    def ensure_loaded(self, name: str) -> bool:
        """确保该模型已在推理服务里（未跑则启动，跑别的则切换）。"""
        self.flog(f"模型请求：{name}")
        ok = self.manager.start(name)
        self.flog(f"模型 {'加载成功' if ok else '加载失败'}：{name}")
        return ok

    def _details(self, entry: dict) -> dict:
        fam = _family_of(entry["name"])
        return {
            "parent_model": "", "format": "gguf", "family": fam,
            "families": [fam],
            "parameter_size": _param_of(entry["name"], entry["size"]),
            "quantization_level": _quant_of(entry["name"]),
        }

    def tags_payload(self) -> dict:
        models = self.manager.list_models()
        out = []
        for m in models:
            out.append({
                "name": m["name"], "model": m["name"],
                "modified_at": _ts(m["mtime"]),
                "size": m["size"],
                "digest": "sha256:" + hashlib.sha256(m["path"].encode("utf-8")).hexdigest(),
                "details": self._details(m),
            })
        return {"models": out}

    def openai_models_payload(self) -> dict:
        """OpenAI /v1/models 格式，返回本地全部模型（不只当前加载的那个）。
        供客户端「读取模型」一次性拿全，避免逐个加载模型才能读到一个。"""
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {"id": m["name"], "object": "model",
                 "created": created, "owned_by": "kaiwu"}
                for m in self.manager.list_models()
            ],
        }

    def ps_payload(self) -> dict:
        cfg = load_config()
        ok, served = self.manager.probe(cfg)
        out = []
        if ok and served:
            for s in served:
                entry = self.resolve_model(s) or {}
                out.append({
                    "name": entry.get("name", s), "model": entry.get("name", s),
                    "size": entry.get("size", 0), "size_vram": entry.get("size", 0),
                    "digest": ("sha256:" + hashlib.sha256(
                        entry.get("path", s).encode("utf-8")).hexdigest()) if entry else "",
                    "expires_at": _now(),
                })
        return {"models": out}


def _ts(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")
