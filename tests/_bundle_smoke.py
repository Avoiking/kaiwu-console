"""_bundle_smoke.py — 整合包冒烟测试：冻结 EXE 的真实首次启动。

测 PyInstaller 产物（dist/kaiwu/kaiwu.exe），不是源码：
  1. 冻结模式能起来（BANNER/网关/WebUI 都打印）
  2. T1a 在 frozen 下生效：app_dir() = EXE 目录（sys.frozen 分支），
     models/ 放假 GGUF → 打"已自动登记 models/ 下的 N 个模型"
  3. frozen 下配置写回路（data/config.json 落在 EXE 旁边）有效
  4. /api/models HTTP 200 且模型卡片在
  5. 干净退出（退出码 0）

BROWSER 指向立即退出的假命令：webbrowser.open 走成功路径（日志行照打），
不在桌面上弹真浏览器。测完清掉 models/ 和 data/，不污染组装。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

DIST = Path(__file__).resolve().parent.parent / "dist" / "kaiwu"
EXE = DIST / "kaiwu.exe"

env = dict(os.environ)
env.pop("KAIWU_LLAMA_SERVER", None)
env["BROWSER"] = "cmd /c exit"

if not EXE.is_file():
    sys.exit(f"❌ 没有 {EXE}，先跑 _make_bundle.py")

# 放假模型（只测登记，不测加载——4KB 假文件）
mdir = DIST / "models" / "TestModel-Q8_0"
mdir.mkdir(parents=True, exist_ok=True)
(mdir / "TestModel-Q8_0.gguf").write_bytes(b"GGUF" + os.urandom(4096))
print(f"假模型就位：{mdir}")

proc = subprocess.Popen([str(EXE)], cwd=DIST, env=env,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1,
                        encoding="utf-8", errors="replace")
lines: list[str] = []


def pump():
    for ln in proc.stdout:
        lines.append(ln.rstrip())
        print("LOG|", ln.rstrip(), flush=True)


threading.Thread(target=pump, daemon=True).start()

# 等冻结 EXE 起来（PyInstaller 冷启动比源码慢）
deadline = time.time() + 40
api_ok = False
models: list = []
while time.time() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/api/models",
                                    timeout=3) as r:
            payload = json.loads(r.read().decode("utf-8"))
            api_ok = True
            models = payload.get("models") or []
            break
    except Exception:
        time.sleep(1)
print(f"\n>>> /api/models: {'✅ 200' if api_ok else '❌ 不通'}，"
      f"模型卡片: {[m.get('name') for m in models]}")

proc.stdin.write("quit\n")
proc.stdin.flush()
time.sleep(1)
proc.stdin.write("\n")
proc.stdin.flush()
try:
    proc.wait(timeout=25)
except Exception:
    proc.kill()
print(f">>> 退出码: {proc.returncode}")

# 冻结写回路：EXE 旁边应落下 data/config.json
cfgp = DIST / "data" / "config.json"
print(f">>> frozen 配置写回: {'✅' if cfgp.is_file() else '❌'}", end=" ")
if cfgp.is_file():
    c = json.loads(cfgp.read_text(encoding="utf-8"))
    print("imported_models =", json.dumps(c.get("imported_models"),
                                          ensure_ascii=False))
else:
    print()

auto = any("自动登记" in l for l in lines)
webui_up = any("Web UI 已启动" in l for l in lines)
browser = any("已在浏览器打开" in l for l in lines)
banner = any("kaiwu 控制台" in l for l in lines)
print(f"\n>>> 冻结启动={bool(banner)} WebUI={webui_up} "
      f"T1a自动登记={auto} 浏览器日志={browser} API={api_ok} "
      f"退出码0={proc.returncode == 0}")

# 清理测试污染（组装只拷 kaiwu.exe + _internal，但顺手保持干净）
shutil.rmtree(DIST / "models", ignore_errors=True)
shutil.rmtree(DIST / "data", ignore_errors=True)

ok = all([banner, webui_up, auto, browser, api_ok, proc.returncode == 0,
          bool(models)])
print(f"\n冒烟测试：{'✅ 通过' if ok else '❌ 有问题，看上面'}")
sys.exit(0 if ok else 1)
