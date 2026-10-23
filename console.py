"""kaiwu 控制台：终端里启动推理服务、切换模型，并内置 Ollama 网关。

用法：python console.py

- list / use / start / stop / status / chat / logs / gateway / quit
- Ollama 网关随控制台启动（默认 11434，被占则退 11437），
  HANA AGENT / ollama CLI / Chatbox 等 Ollama 客户端可直接接入，
  请求哪个模型就自动切换加载哪个模型。
"""
from __future__ import annotations

import os
import sys

# ---- DPI 感知：必须在这里、在所有窗口创建之前声明 -------------------------
# Python 进程默认是 DPI-UNAWARE，系统会把我们的窗口画好后整体拉伸 ——
# 在高分屏（如 2560×1600 @100% 缩放）上就表现成"字糊"。
# 资源管理器清晰是因为它自带 SYSTEM_AWARE。
# ⚠️ 顺序有讲究：import tkinter / 创建任何窗口之前就要设，之后就锁定了。
if sys.platform == "win32":
    try:
        import ctypes
        # 2 = PER_MONITOR_AWARE_V2 语义（Win8.1+）；失败退回老 API
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import requests

from app.config import app_dir, load_config
from app.kaiwu_manager import KaiwuManager
from app.ollama_gateway import OllamaGateway
from app.webui import WebUI

DIM, RESET = "\033[90m", "\033[0m"

HELP = """命令列表：
  list | ls          列出本地可用模型（* 为当前模型）
  use <名称|序号>     切换模型：卸载当前模型并加载新模型
  measure <名称|序号> 测量该模型能装下的最大上下文（只需跑一次，结果会记住）
  start              启动服务（用上次/首个模型）
  stop               停止推理服务（释放显存）
  status             服务与网关状态
  chat               进入对话（:q 退出，Ctrl+C 打断生成）
  logs               查看推理服务最近日志
  gateway            网关接入信息
  ui                 显示 Web UI 网址（浏览器版控制台，已随控制台自启）
  quit | exit | q    退出控制台
"""

BANNER = """
======================================================
  kaiwu 控制台 —— 本地模型推理 + Ollama 网关
======================================================
输入 help 查看命令；use <模型> 切换模型；chat 直接对话
"""


def _gb(size: int) -> str:
    return f"{size / (1024 ** 3):.1f}GB"


def _auto_register_models_dir(webui: "WebUI", cfg: dict) -> None:
    """首启自动登记程序目录下的 models/ —— 整合包"解压即用"的关键一步。

    2026-10-17 假新机测试实证：list_models 的扫描根目录只来自
    imported_models / model_units 登记表，import_dir 不参与扫描 —— 新机器
    把 GGUF 丢进 models/ 后，不手动点"添加模型文件夹"的话卡片上一个模型
    都不显示，"解压 → 丢模型 → 双击 → 开聊"的承诺直接破产。名册为空时
    自动扫一遍，把这一步补上。

    两条纪律（整合包计划定）：
      · 只在名册为空时做 —— 不覆盖用户手动添加的其它目录；
      · models/ 不存在或没有 .gguf → 安静跳过，不报错不打扰。

    返回（可能重载过的）cfg：models_add 写的是磁盘，调用方手里那份是启动时
    的旧快照 —— 不还回去的话，紧接着的首次列模型会打"没有找到本地模型"，
    而网页卡片明明有（API 每次重读配置）。
    """
    if cfg.get("imported_models"):
        return cfg
    mdir = app_dir() / "models"
    try:
        if not mdir.is_dir() or not any(mdir.rglob("*.gguf")):
            return cfg
    except OSError:
        return cfg
    res = webui.models_add(str(mdir))
    if res.get("ok"):
        n = len(res.get("added") or [])
        sk = len(res.get("skipped") or [])
        msg = f"已自动登记 models/ 下的 {n} 个模型"
        if sk:
            msg += f"（{sk} 个已在名册，跳过）"
        print(msg)
        for note in res.get("notes") or []:
            print(f"  [提示] {note}")
        return load_config()
    print(f"[提示] models/ 自动登记跳过：{res.get('error')}")
    return cfg


def cmd_list(manager: KaiwuManager, cfg: dict) -> None:
    models = manager.list_models(cfg)
    if not models:
        print("没有找到本地模型：把 .gguf 放进权重目录或检查 config.imported_models")
        return
    ok, served = manager.probe(cfg)
    print(f"{'共 ' + str(len(models)) + ' 个模型':<}")
    for i, m in enumerate(models, 1):
        cur = "*" if ok and manager._model_matches(served, m["name"]) else " "
        vis = " [视觉]" if KaiwuManager.has_vision(cfg, m["name"], m.get("path", "")) else ""
        src = "" if m["imported"] else "  (扫描发现)"
        print(f" {cur}{i:>2}) {m['name']:<32} {_gb(m['size']):>7}{vis}{src}")


def _pick(manager: KaiwuManager, cfg: dict, arg: str) -> dict | None:
    models = manager.list_models(cfg)
    if arg.isdigit():
        i = int(arg) - 1
        if 0 <= i < len(models):
            return models[i]
        print(f"序号超出范围（1-{len(models)}），先 list 查看")
        return None
    al = arg.lower()
    for m in models:
        if m["name"].lower() == al:
            return m
    for m in models:
        if al in m["name"].lower():
            return m
    if arg.lower().endswith(".gguf") and os.path.isfile(arg):
        return {"name": arg, "path": arg, "size": 0, "mtime": 0, "imported": False}
    print(f"找不到模型 {arg!r}（list 查看；也可 use <完整gguf路径>）")
    return None


def cmd_use(manager: KaiwuManager, cfg: dict, arg: str) -> None:
    if not arg:
        print("用法：use <模型名|序号>（先 list 查看）")
        return
    target = _pick(manager, cfg, arg)
    if target is None:
        return
    print(f"→ {target['name']}")
    if manager.start(target["name"], cfg):
        print(f"[OK] 已就绪：{target['name']}")
    else:
        print(f"[失败] {manager.last_error}")


def cmd_measure(manager: KaiwuManager, cfg: dict, arg: str) -> None:
    """自动测量某模型能装下的最大上下文（每槽），结果按「模型@量化档」
    缓存到 config.ctx_measured。

    只需跑一次：结果连同环境指纹（显卡/显存/KV量化/llama 版本）一起写进配置，
    之后每次加载都直接采用。换了显卡或改了 KV 量化，指纹对不上会自动重测；
    切换到没测过的量化档，也会各自测一次（体积不同，窗口不同）。

    流程：一次温和加载读出显存账目 → 算式估算 → 加载验证 → 少量二分兜底。
    """
    if not arg:
        print("用法：measure <模型名|序号>")
        print("  测出该模型能装下的最大上下文并记住。通常 2~3 次加载、几十秒。")
        return
    target = _pick(manager, cfg, arg)
    if target is None:
        return
    name = target["name"]
    print(f"→ 测量 {name}")
    print("  首次会真实加载模型几次，之后永久沿用此结果…")
    res = manager.measure_ctx(name, cfg)
    print()
    for a in res["attempts"]:
        mark = "OK " if a["ok"] else "失败"
        detail = f"  GPU {a['on_gpu']}/{a['total']} 层" if a["ok"] else ""
        print(f"  [{mark}] 请求 {a['requested']}{detail}")
    if res["best"]:
        loads = res.get("loads", len(res["attempts"]))
        print(f"\n[OK] {name} 最大可用上下文 = {res['best']}（每槽）")
        print(f"     共 {loads} 次加载，已写入 config.ctx_measured（模型@量化档）+ 环境指纹")
        print("     以后加载该量化档自动采用（换显卡/改 KV 量化/换量化档会自动重测）")
    else:
        print(f"\n[失败] {manager.last_error}")


def cmd_start(manager: KaiwuManager, cfg: dict) -> None:
    model = str(cfg.get("last_model") or "")
    if not model:
        models = manager.list_models(cfg)
        if not models:
            print("没有可用模型")
            return
        model = models[0]["name"]
    if manager.start(model, cfg):
        print(f"[OK] 已就绪：{model}")
    else:
        print(f"[失败] {manager.last_error}")


def cmd_stop(manager: KaiwuManager, cfg: dict) -> None:
    if manager.stop(cfg):
        print("[OK] 推理服务已停止，显存已释放")
    else:
        print(f"[失败] {manager.last_error}")


def cmd_status(manager: KaiwuManager, gateway: OllamaGateway, cfg: dict) -> None:
    st = manager.status(cfg)
    names = {"ready": "运行中", "starting": "启动中", "stopped": "已停止"}
    print(f"推理服务：{names.get(st['state'], st['state'])}")
    if st["endpoint_alive"]:
        print(f"  OpenAI 端点：{st['base_url']}（已加载：{', '.join(st['serving']) or '?'}）")
    else:
        print(f"  OpenAI 端点：{st['base_url']}（无响应）")
    print(f"  目标模型：{st['model'] or '未设置'}")
    if gateway.running:
        print(f"Ollama 网关：http://127.0.0.1:{gateway.port}（运行中）")
    else:
        print("Ollama 网关：未运行")


def cmd_gateway(manager: KaiwuManager, gateway: OllamaGateway, cfg: dict) -> None:
    print(f"OpenAI 直连（推荐，HANA AGENT 用这个）：{manager.base_url(cfg)}/v1")
    print(f"  - 模型名随意填（llama-server 不校验），实际用控制台加载的模型")
    print(f"  - 换模型：回控制台 use <模型>（直连不自动切换）")
    if gateway.running:
        print(f"Ollama 网关（备用）：http://127.0.0.1:{gateway.port}")
        print(f"  - ollama 命令行：set OLLAMA_HOST=http://127.0.0.1:{gateway.port} 后 ollama list / ollama run <模型名>")
        print(f"  - 请求哪个模型自动切换哪个（适合 Chatbox 等不方便手动切的客户端）")
    else:
        print("Ollama 网关：未运行")


def cmd_logs(manager: KaiwuManager) -> None:
    lines = manager.log_text().splitlines()
    for line in lines[-40:]:
        print(f"  {line}")
    if not lines:
        print("  （暂无日志）")


# ------------------------------------------------------------ 对话
def _stream_chat(base_url: str, model: str, messages: list[dict]):
    """生成器：yield ("think"|"text"|"error", str)。"""
    resp = None
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={"model": model, "messages": messages, "stream": True},
            stream=True, timeout=(10, 3600),
            headers={"Accept": "text/event-stream"},
        )
        if resp.status_code != 200:
            yield ("error", f"HTTP {resp.status_code}: {resp.text[:300]}")
            return
        resp.encoding = "utf-8"  # SSE 头无 charset，防中文乱码
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            import json
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if obj.get("error"):
                yield ("error", str(obj["error"]))
                return
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("reasoning_content"):
                yield ("think", delta["reasoning_content"])
            if delta.get("content"):
                yield ("text", delta["content"])
    except Exception as e:
        yield ("error", str(e))
    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass


def chat_mode(manager: KaiwuManager, cfg: dict) -> None:
    state, model = manager.state()
    if state != "ready":
        model = str(cfg.get("last_model") or "")
        if not model:
            models = manager.list_models(cfg)
            if models:
                model = models[0]["name"]
        if not model or not manager.start(model, cfg):
            print("  服务未就绪：先 use <模型> 加载模型")
            return
    print(f"对话模型：{model}（:q 退出，Ctrl+C 打断生成）")
    history: list[dict] = []
    while True:
        try:
            line = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in (":q", "/q", "exit", "quit"):
            break
        history.append({"role": "user", "content": line})
        print("\nkaiwu> ", end="", flush=True)
        full: list[str] = []
        err = ""
        in_think = False
        try:
            for kind, text in _stream_chat(manager.base_url(cfg), model, history):
                if kind == "error":
                    err = text
                    break
                if kind == "think":
                    if not in_think:
                        print(f"{DIM}[思考] ", end="", flush=True)
                        in_think = True
                    print(text, end="", flush=True)
                else:
                    if in_think:
                        print(f"{RESET}\n", end="", flush=True)
                        in_think = False
                    print(text, end="", flush=True)
                    full.append(text)
        except KeyboardInterrupt:
            print(f"{RESET}\n  [已打断]")
        print(RESET)
        if err:
            print(f"  [错误] {err}")
            history.pop()
            continue
        if full:
            history.append({"role": "assistant", "content": "".join(full)})
        else:
            history.pop()
            print("  [无正文输出，本轮不计入历史]")
        print(flush=True)


# ------------------------------------------------------------ 主循环
def main() -> None:
    os.system("")  # 启用 Windows 控制台 ANSI 转义
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    cfg = load_config()
    manager = KaiwuManager(progress=lambda line: print(f"  {line}", flush=True))
    gateway = OllamaGateway(manager, log=print)
    webui = WebUI(manager, gateway=gateway, log=print)

    print(BANNER)
    try:
        gateway.start()
    except RuntimeError as e:
        print(f"[警告] 网关启动失败：{e}")
    try:
        webui.start()
    except RuntimeError as e:
        print(f"[警告] Web UI 启动失败：{e}")

    cfg = _auto_register_models_dir(webui, cfg)

    # T1b：自动打开浏览器（整合包双击即用的最后一块；打不开不碍事，
    # URL 上面已经打印过）。配置 auto_open_browser=false 可关。
    if webui.running and cfg.get("auto_open_browser", True):
        try:
            import webbrowser
            webbrowser.open(f"http://127.0.0.1:{webui.port}")
            print(f"已在浏览器打开控制台：http://127.0.0.1:{webui.port}"
                  f"（打不开就手动访问）")
        except Exception:
            pass

    ok, served = manager.probe(cfg)
    if ok:
        print(f"推理服务已在运行：{', '.join(served)} @ {manager.base_url(cfg)}")
    else:
        print(f"推理服务未运行：use <模型> 选择模型，或 start 用上次的模型")
    print()
    cmd_list(manager, cfg)

    while True:
        try:
            line = input("\nkaiwu> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            break
        elif cmd in ("help", "h", "?"):
            print(HELP)
        elif cmd in ("list", "ls", "models"):
            cmd_list(manager, cfg)
        elif cmd == "use":
            cmd_use(manager, cfg, arg)
        elif cmd == "measure":
            cmd_measure(manager, cfg, arg)
        elif cmd == "start":
            cmd_start(manager, cfg)
        elif cmd == "stop":
            cmd_stop(manager, cfg)
        elif cmd == "status":
            cmd_status(manager, gateway, cfg)
        elif cmd == "gateway":
            cmd_gateway(manager, gateway, cfg)
        elif cmd == "ui":
            print(f"Web UI：http://127.0.0.1:{webui.port}" if webui.running
                  else "Web UI：未运行")
        elif cmd == "logs":
            cmd_logs(manager)
        elif cmd == "chat":
            chat_mode(manager, cfg)
        else:
            print(f"未知命令 {cmd!r}，help 查看")

    try:
        ans = input("退出前停止推理服务吗？(y=停止 / 回车保留运行): ").strip().lower()
    except Exception:
        ans = ""
    webui.stop()
    if ans == "y":
        manager.stop(cfg)
        print("[OK] 推理服务已停止")
    else:
        print(f"推理服务保留运行：{manager.base_url(cfg)}（OpenAI 直连可用；"
              f"Ollama 网关随控制台退出，下次启动控制台即恢复）")
    print("再见。")


if __name__ == "__main__":
    main()
