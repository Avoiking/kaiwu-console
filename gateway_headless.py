"""kaiwu 网关无头模式：供服务编排脚本（OCV manage_services.ps1）拉起。

等效 console.py 的网关部分，但无交互界面、常驻后台：
- Ollama 协议网关（默认 11434，被占则退 11437）
- 请求哪个模型自动加载哪个（KaiwuManager.start）
- keep_alive=0 卸载、think:false 由网关翻译成 llama-server 语义

用法（工作目录必须是本文件所在目录，app 包按相对导入解析）：
  cd D:\\kaiwu && python gateway_headless.py
"""
from __future__ import annotations

import sys
import time

# DPI 感知：与 console.py 一致。网关自己不弹窗，但若将来接了带界面的
# 子功能（如选文件夹），提前声明可避免"窗口被系统拉伸"导致的发虚。
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

from app.kaiwu_manager import KaiwuManager
from app.ollama_gateway import OllamaGateway


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    manager = KaiwuManager(progress=lambda line: print(f"  {line}", flush=True))
    gateway = OllamaGateway(manager, log=lambda s: print(f"[gateway] {s}", flush=True))
    port = gateway.start()
    print(f"HEADLESS_GATEWAY_READY port={port}", flush=True)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
