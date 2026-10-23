"""_make_bundle.py — kaiwu 新手整合包一键构建（T2 干净配置 + T3 PyInstaller + T4 组装）。

用法（在仓库根目录，用构建 venv 的 Python 跑）：
    .build_venv/Scripts/python.exe tests/_make_bundle.py
    .build_venv/Scripts/python.exe tests/_make_bundle.py --skip-build   # 只重新组装

产物：<仓库>/dist-bundle/ —— 整个文件夹就是整合包
    kaiwu.exe + _internal/   ← PyInstaller onedir（console.py 入口，含 requests）
    bin/                     ← llama-server.exe + 全套 DLL（CUDA/cuBLAS/OpenSSL）
    data/config.json         ← 干净配置：DEFAULTS + 显式清空全部本机键（§2 红线）
    models/                  ← 空目录 + 说明；用户丢 GGUF，首启自动登记（T1a）
    必读.txt                 ← 三行字级别说明

llama-server 来源解析（不硬编码用户名，按优先级）：
    1. --llama-src 参数
    2. 本机 data/config.json 的 llama_server_exe 所在目录
    3. 环境变量 KAIWU_LLAMA_SERVER
    4. ~/.kaiwu/bin* 下含 llama-server.exe 的最新目录
OpenSSL DLL 来源：仓库 llama-deps/（gitignore 的本机依赖目录）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".build_venv" / "Scripts" / "python.exe"
DIST = ROOT / "dist"
WORK = ROOT / "build"
BUNDLE = ROOT / "dist-bundle"
LLAMA_DEPS = ROOT / "llama-deps"

# §2 红线：干净配置里这些键必须是空/空字典（本机私人数据，绝不进包）
CLEAN_STR_KEYS = ("tavily_key", "llama_server_exe", "ollama_dir", "import_dir",
                  "last_model")
CLEAN_DICT_KEYS = ("imported_models", "model_units", "model_variants",
                   "mmproj_models", "draft_models", "ctx_measured",
                   "ctx_measured_env", "ctx_measured_note", "attach_overrides",
                   "moe_offload", "model_ctx_overrides")

README_TXT = """kaiwu 新手整合包 —— 解压即用，不需要装 Python

使用步骤：
  1. 放模型：下载 GGUF 模型（huggingface.co 搜"GGUF"），一个模型一个文件夹，
     整个文件夹放进本目录的 models/ 里。
     例：models/Ornith-1.5-9B/Ornith-1.5-9B-Q4_K_M.gguf
  2. 双击 kaiwu.exe，浏览器会自动打开控制台（几秒）；没弹就手动访问
     http://127.0.0.1:8080
  3. 第一次加载某个模型时会自动测量上下文（要真实加载几次，等一两分钟），
     之后直接聊天。嫌窗口小/速度低，模型卡片上可以手动调上下文。

说明：
  · 配置和测量记录在本目录 data/ 里，删掉即重置
  · 显存不够时，大模型会自动把一部分权重放内存，速度会降——卡片备注有写清楚
  · 纯本地软件，模型和数据都不出本机

如果杀软 / 系统拦截（未签名小工具的常态，不是病毒）：
  · Windows SmartScreen 弹「Windows 已保护你的电脑」：点提示框里的
    「更多信息」→「仍要运行」即可
  · Defender / 火绒等报 llama-server.exe（bin/ 里的推理引擎）：在杀软的
    隔离/保护历史记录里选「还原」+「允许」；它是 llama.cpp 官方版二进制，
    只是没买数字签名
"""

MODELS_TXT = """把模型文件夹放进这个目录。约定：一个模型一个文件夹，.gguf 直接放在文件夹里。

例：models/Ornith-1.5-9B/Ornith-1.5-9B-Q4_K_M.gguf

第一次启动 kaiwu.exe 会自动登记这里的所有模型；
之后新加的模型，在控制台网页里点「＋ 添加模型文件夹」再选一次本目录即可。
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def _resolve_llama_src(arg: str) -> Path:
    """llama-server 来源目录（含全套 DLL 的那个目录）。"""
    if arg:
        p = Path(arg)
        if (p / "llama-server.exe").is_file():
            return p
        raise SystemExit(f"--llama-src 指向的目录里没有 llama-server.exe：{p}")
    cfgp = ROOT / "data" / "config.json"
    if cfgp.is_file():
        try:
            exe = json.loads(cfgp.read_text(encoding="utf-8")).get("llama_server_exe")
            if exe and Path(exe).is_file():
                return Path(exe).parent
        except Exception:
            pass
    env = __import__("os").environ.get("KAIWU_LLAMA_SERVER", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p.parent
        if (p / "llama-server.exe").is_file():
            return p
    cands = sorted(Path.home().glob(".kaiwu/bin*"),
                   key=lambda p: p.name, reverse=True)
    for c in cands:
        if (c / "llama-server.exe").is_file():
            return c
    raise SystemExit("找不到 llama-server 来源目录：用 --llama-src 指定，"
                     "或在本机 data/config.json 配好 llama_server_exe")


def clean_config() -> dict:
    """T2：从 DEFAULTS 生成干净配置，再显式清空全部本机键（双保险）。"""
    sys.path.insert(0, str(ROOT))
    from app.config import DEFAULTS  # noqa: E402
    cfg = dict(DEFAULTS)
    for k in CLEAN_STR_KEYS:
        cfg[k] = ""
    for k in CLEAN_DICT_KEYS:
        cfg[k] = {}
    return cfg


def build() -> None:
    """T3：PyInstaller onedir 构建 console.py。"""
    if not VENV_PY.is_file():
        raise SystemExit(f"没有构建 venv：{VENV_PY}\n"
                         "先建：.build_venv 不存在时用 managed python 建 venv 并 "
                         "pip install pyinstaller requests（见打包计划 T3）")
    cmd = [str(VENV_PY), "-m", "PyInstaller",
           "--onedir", "--clean", "--noconfirm",
           "--name", "kaiwu",
           "--icon", str(ROOT / "app" / "kaiwu.ico"),
           "--distpath", str(DIST), "--workpath", str(WORK),
           "--specpath", str(WORK),
           str(ROOT / "console.py")]
    log("== T3 PyInstaller 构建（可能要几分钟）==")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"PyInstaller 失败（码 {r.returncode}）")
    log(f"   构建完成（{time.time() - t0:.0f}s）→ {DIST / 'kaiwu'}")


def assemble(llama_src: Path) -> None:
    """T4：组装 dist-bundle/。

    ⚠️ 原地更新，不整目录 rmtree 重建（2026-10-17 改）：安全钩子会拦下
    批量删除（上次重建被挡：131 个文件超阈值），且 bin/ 是 1.1G 的
    llama-server 全套——代码改动重建时它一个字节都没变，重拷纯浪费。
    做法：kaiwu.exe + _internal/ 覆盖更新，bin/ 缺才拷，config/models/
    必读 每次重写。文件集在同 PyInstaller + 同依赖下稳定，覆盖即最新。
    """
    exe = DIST / "kaiwu" / "kaiwu.exe"
    internal = DIST / "kaiwu" / "_internal"
    if not exe.is_file() or not internal.is_dir():
        raise SystemExit(f"PyInstaller 产物不完整：{DIST / 'kaiwu'}")

    BUNDLE.mkdir(parents=True, exist_ok=True)

    log("== T4 组装（原地更新）==")
    shutil.copy2(exe, BUNDLE / "kaiwu.exe")
    shutil.copytree(internal, BUNDLE / "_internal", dirs_exist_ok=True)
    log(f"   kaiwu.exe + _internal/ ← {DIST / 'kaiwu'}")

    # bin/ ← llama-server 全套 + OpenSSL DLL（放 exe 旁边，_spawn_env 会前置 PATH）。
    # 已在位就跳过：它不随代码变动，1.1G 没必要每次重拷。
    bindir = BUNDLE / "bin"
    if not (bindir / "llama-server.exe").is_file():
        shutil.copytree(llama_src, bindir)
        if LLAMA_DEPS.is_dir():
            for dll in LLAMA_DEPS.glob("*.dll"):
                shutil.copy2(dll, bindir / dll.name)
        n = len(list(bindir.iterdir()))
        log(f"   bin/ ← {llama_src}（{n} 个文件，含 OpenSSL DLL）")
    else:
        log(f"   bin/ 已在位（{len(list(bindir.iterdir()))} 个文件），跳过拷贝")

    # data/config.json ← 干净配置
    datadir = BUNDLE / "data"
    datadir.mkdir(exist_ok=True)
    cfg = clean_config()
    (datadir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    log("   data/config.json ← DEFAULTS 干净配置（本机键已显式清空）")

    # 运行期产物清场：日志和聊天记录是**分发前必须清掉的个人数据**
    # （2026-10-17 实测：本机测过的包装着 gateway.log + webui_sessions/
    #   里一条真实聊天记录 + config 里的 D:/ 模型路径和测量记录）
    logf = datadir / "gateway.log"
    if logf.is_file():
        logf.unlink()
        log("   data/gateway.log ← 已清（运行日志，含本机路径）")
    sess = datadir / "webui_sessions"
    if sess.is_dir():
        shutil.rmtree(sess)
        log("   data/webui_sessions/ ← 已清（聊天记录）")

    # models/ ← 空目录 + 说明（T1a 首启自动登记）
    models_dir = BUNDLE / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "把模型放这里.txt").write_text(MODELS_TXT, encoding="utf-8")
    log("   models/ ← 空目录 + 说明（首启自动登记，T1a）")

    (BUNDLE / "必读.txt").write_text(README_TXT, encoding="utf-8")
    log("   必读.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true",
                    help="跳过 PyInstaller，只用现有 dist/kaiwu 重新组装")
    ap.add_argument("--llama-src", default="",
                    help="llama-server 来源目录（默认自动解析）")
    args = ap.parse_args()

    llama_src = _resolve_llama_src(args.llama_src)
    log(f"llama-server 来源：{llama_src}")

    if not args.skip_build:
        build()
    assemble(llama_src)

    total = sum(f.stat().st_size for f in BUNDLE.rglob("*") if f.is_file())
    log(f"\n✅ 整合包就绪：{BUNDLE}")
    log(f"   总大小 {total / 1024 ** 3:.2f} GB")
    log("   分发：把 dist-bundle 整个文件夹改名/压缩即可（隐私验收跑 "
        "_bundle_check.py）")


if __name__ == "__main__":
    main()
