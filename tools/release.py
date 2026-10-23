"""tools/release.py — 一键发版：把当前工作区打成干净树 → 扫描隐私 → 提交 → 推送。

为什么要有这个脚本
------------------
公开仓库（GitHub 上那个）只有一个提交 —— 干净树，无任何本机信息。
本地工作区是开发用，两者内容会随着开发 diverge。发版时要把「工作区的当前状态」
变成「公开仓库的单个干净提交」，中间要做：导出 → 隐私扫描 → 提交 → 推。
手工做容易漏步骤。

用法
----
    python tools/release.py check     # 只扫隐私，不改任何东西
    python tools/release.py publish   # 扫隐私 → 提交 → 推送（main + tag）
    python tools/release.py --stage-only   # 只更新暂存目录，不提交不推
    python tools/release.py publish --stage-dir D:\\somewhere\\else

暂存目录默认取环境变量 KAIWU_RELEASE_STAGE，没有就用系统临时目录下的
kaiwu-release-stage（不写死在代码里，免得泄露本机路径）。

隐私扫描的判定
--------------
RED_PATTERNS 是**必须零命中**的（泄露即失败）。
本机专属模式（你的用户名、模型目录名…）从 .githooks/local-patterns.txt 读，
那个文件不进版本库 —— 所以这个脚本本身可以公开，谁都能用来验自己的树。

退出码：0 = 通过；1 = 有红线命中。
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 公开仓库地址（可用 --remote 覆盖）
REMOTE = os.environ.get(
    "KAIWU_RELEASE_REMOTE", "git@github.com:Avoiking/kaiwu-console.git")

# ============================================================
# 公开仓库只放「开源必要的东西」
# ============================================================
# 用户 2026-10-23 定：内部资料（验收报告 / 竞争分析 / 交接归档 /
# 对比报告 HTML / 推送守卫）**一律不外传** —— 那是给我们自己看的。
# 保留：源码 + 测试 + 发版脚本 + CI + LICENSE + README + 最小启动入口。
KEEP_DIRS = ["app", "tests", "tools", ".github"]
KEEP_FILES = [
    "console.py",
    "gateway_headless.py",
    "requirements.txt",
    "config.example.json",
    "LICENSE",
    "README.md",
    "启动控制台.bat",
    ".gitignore",
]

# 除 git 跟踪的文件外，额外要带上的（打包三件套）
EXTRA = [
    ".github/workflows/ci.yml",
    "tests/_make_bundle.py",
    "tests/_bundle_check.py",
    "tests/_bundle_smoke.py",
    "tools/release.py",
]

# 明确**不**上传的（防御性清单：万一以后不小心 git add 了也会被这里挡住）
DROP = [
    "docs/",                       # 内部验收报告 / 竞争分析 / 交接归档
    "kaiwu-vs-ollama/",            # 对比报告 HTML（内部用）
    ".githooks/",                  # 推送守卫（个人工作流）
    "MoE专家分配优化_实施方案.md",  # 历史方案文档（内部）
    "用户指南_模型管理.md",          # 内部使用手册
    "HANDOFF.md",
]

# 必须零命中的模式
RED_PATTERNS = {
    "Windows 用户名": r"min[i]BA",
    "本机模型目录名": r"kaiwu[-/]models",
    "绝对家目录路径": r"[A-Z]:[\\/]Users[\\/][A-Za-z0-9_.-]+",
    "显卡型号": r"RTX \d+",
    "仓库绝对路径": r"[A-Z]:[\\/]kaiwu",
    "真实密钥形态": (r"tvly-[A-Za-z0-9_-]{16,}"
                    r"|sk-[A-Za-z0-9_-]{20,}"
                    r"|ghp_[A-Za-z0-9]{30,}"),
}

# 信息级：出现不拦截（通用正则模式、工具名）
INFO_PATTERNS = {
    "本地工具目录名": r"workbuddy",
}

# 本机专属模式文件（不进版本库）
LOCAL_PAT = ROOT / ".githooks" / "local-patterns.txt"


def local_patterns() -> list[str]:
    """从 local-patterns.txt 读本机专属红线（按正则用，能匹配 min[i]BA）。"""
    if not LOCAL_PAT.is_file():
        return []
    out = []
    for line in LOCAL_PAT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          check=True).stdout.decode("utf-8", errors="replace")


def default_stage() -> Path:
    """暂存目录：环境变量 > 系统临时目录。不写死本机路径。"""
    env = os.environ.get("KAIWU_RELEASE_STAGE", "").strip()
    if env:
        return Path(env)
    return Path(tempfile.gettempdir()) / "kaiwu-release-stage"


def _dropped(rel: str) -> bool:
    """这个路径是不是明确不外传的？"""
    for d in DROP:
        if rel == d or rel.startswith(d) or d.endswith("/") and rel.startswith(d):
            return True
    return False


def stage(stage_dir: Path) -> int:
    """把工作区的干净树导出到暂存目录（只导白名单里的东西）。"""
    if stage_dir.exists():
        for p in stage_dir.rglob("*"):
            if p.is_file():
                try:
                    p.chmod(0o666)
                except OSError:
                    pass
        shutil.rmtree(stage_dir, ignore_errors=True)
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    tracked = [p for p in git("ls-files", "-z").split("\0") if p]
    # 白名单过滤：只留 KEEP_DIRS 下的 + KEEP_FILES 列出的 + EXTRA
    keep = []
    for rel in tracked + EXTRA:
        if rel in keep:
            continue
        if _dropped(rel):
            continue
        top = rel.split("/")[0]
        if top in KEEP_DIRS or rel in KEEP_FILES or rel in EXTRA:
            keep.append(rel)

    n = 0
    for rel in keep:
        src = ROOT / rel
        if not src.is_file():
            print(f"跳过（不存在）: {rel}")
            continue
        dst = stage_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    print(f"同步 {n} 个文件 → {stage_dir}")
    return 0


def scan(stage_dir: Path) -> int:
    """扫暂存目录，红线必须零命中。"""
    if not stage_dir.is_dir():
        print(f"❌ 没有暂存目录 {stage_dir}，先跑 stage")
        return 1

    red = dict(RED_PATTERNS)
    for p in local_patterns():
        red[f"本机专属: {p}"] = p

    bad = 0
    for label, pat in red.items():
        rx = re.compile(pat)
        hits = []
        for f in stage_dir.rglob("*"):
            if not f.is_file():
                continue
            try:
                text = f.read_bytes().decode("utf-8", errors="ignore")
            except OSError:
                continue
            m = rx.search(text)
            if m:
                hits.append(f"{f.relative_to(stage_dir)}  ::  {m.group(0)!r}")
        if hits:
            bad += 1
            print(f"❌ {label}: {len(hits)} 处")
            for h in hits[:10]:
                print(f"     {h}")
        else:
            print(f"✅ {label}: 零命中")

    for label, pat in INFO_PATTERNS.items():
        rx = re.compile(pat)
        hits = [f for f in stage_dir.rglob("*")
                if f.is_file()
                and rx.search(f.read_bytes().decode("utf-8", errors="ignore"))]
        if hits:
            print(f"ℹ️  {label}: {len(hits)} 处（信息级，不拦截）")

    print()
    if bad:
        print(f"❌ 有 {bad} 类红线命中，不能发布")
        return 1
    print("✅ 全部红线零命中，可以发布")
    return 0


AUTHOR_NAME = "Avoiking"
AUTHOR_EMAIL = "Avoiking@users.noreply.github.com"


def git_env() -> dict:
    """git 子进程的环境变量。

    ⚠️ 必须显式盖掉 GIT_COMMITTER_NAME / GIT_COMMITTER_EMAIL：
    某些沙箱环境会预设这两个变量（如 trae-solo-agent），
    **环境变量优先级高于 git config**，结果就是提交记录里
    署名变成「Avoiking and trae-solo-agent」——作者是本人、
    提交者是工具自己。对外发布的仓库不该带这种痕迹。
    """
    env = dict(os.environ)
    env["GIT_COMMITTER_NAME"] = AUTHOR_NAME
    env["GIT_COMMITTER_EMAIL"] = AUTHOR_EMAIL
    env["GIT_AUTHOR_NAME"] = AUTHOR_NAME
    env["GIT_AUTHOR_EMAIL"] = AUTHOR_EMAIL
    return env


def commit_and_push(stage_dir: Path, tag: str) -> int:
    """在暂存目录里建仓 → 单次提交 → 推 main 和 tag。"""
    env = git_env()
    if not (stage_dir / ".git").is_dir():
        subprocess.run(["git", "init", "-q"], cwd=stage_dir, check=True, env=env)
    for k, v in (("user.name", AUTHOR_NAME), ("user.email", AUTHOR_EMAIL)):
        subprocess.run(["git", "config", k, v], cwd=stage_dir, check=True, env=env)
    # 远端：默认用公开仓库地址，可用 --remote 覆盖
    remotes = subprocess.run(["git", "remote"], cwd=stage_dir, env=env,
                             capture_output=True, text=True).stdout.split()
    if "origin" not in remotes:
        subprocess.run(["git", "remote", "add", "origin", REMOTE],
                       cwd=stage_dir, check=True, env=env)
    # 统一用 main（git init 的默认分支名可能是 master）
    cur = subprocess.run(["git", "symbolic-ref", "--short", "HEAD"],
                         cwd=stage_dir, env=env,
                         capture_output=True, text=True).stdout.strip()
    if cur and cur != "main":
        subprocess.run(["git", "branch", "-m", "main"], cwd=stage_dir, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=stage_dir, check=True, env=env)

    msg = (
        f"kaiwu 控制台 {tag}：本地大模型推理控制台\n"
        "\n"
        "Web UI 聊天 + Ollama 协议网关 + OpenAI 兼容直连，一个 Python 进程搞定。\n"
        "\n"
        "核心能力：\n"
        "- 上下文按模型实际能力测量并存档，跨环境指纹自动作废重测\n"
        "- MoE 模型沿 llama-server --fit-target 显存预留档位逐档真加载实测，\n"
        "  自动择优出最快的档位（先算后测：数学剪枝 + 实测裁判）\n"
        "- 中文界面、零云端依赖、权重全在本机\n"
        "\n"
        "发布前做过两轮隐私验收：整合包 T5 红线零命中、仓库全量扫描零命中。\n"
    )
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=stage_dir,
                   check=True, env=env)
    subprocess.run(["git", "tag", "-f", tag], cwd=stage_dir, check=True, env=env)

    log = subprocess.run(["git", "log", "--oneline"], cwd=stage_dir, env=env,
                         capture_output=True, text=True).stdout.strip()
    cnt = subprocess.run(["git", "ls-files"], cwd=stage_dir, env=env,
                         capture_output=True, text=True).stdout.count("\n")
    print(log)
    print(f"文件数: {cnt}")

    print("\n== 推送 ==")
    r = subprocess.run(["git", "push", "--force", "origin", "main"],
                       cwd=stage_dir, env=env, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        return 1
    r = subprocess.run(["git", "push", "--force", "origin", tag],
                       cwd=stage_dir, env=env, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "publish", "stage-only"])
    ap.add_argument("--tag", default="v0.1.0-preview")
    ap.add_argument("--stage-dir", default="")
    args = ap.parse_args()

    stage_dir = Path(args.stage_dir) if args.stage_dir else default_stage()

    print("== 1. 导出干净树 ==")
    stage(stage_dir)
    print()
    print("== 2. 隐私扫描 ==")
    if scan(stage_dir) != 0:
        return 1
    if args.cmd in ("check", "stage-only"):
        return 0
    print()
    print(f"== 3. 提交并推送（tag={args.tag}）==")
    return commit_and_push(stage_dir, args.tag)


if __name__ == "__main__":
    sys.exit(main())
