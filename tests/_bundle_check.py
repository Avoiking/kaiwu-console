"""_bundle_check.py — T5 整合包隐私验收（打包计划 §2 红线，硬性门槛）。

对 dist-bundle/ 全文件（二进制安全）递归搜索个人模式，必须零命中。

通用红线（写死在脚本里，谁都能用）：
    tvly-        Tavily API key 前缀
    C:\\Users\\<本机用户名>  绝对家目录路径（含用户名，按当前机器动态生成）

本机专属红线（从 .githooks/local-patterns.txt 读，那个文件不进版本库）：
    Windows 用户名、模型目录名、显卡型号等 —— 各人自己填，脚本本身可公开。

另做两项专项：
  1. data/config.json：本机键必须全空（tavily_key/imported_models/llama_server_exe 等）
  2. data/ 内的 .kaiwu：必须零命中（自家源码合法引用 ~/.kaiwu 约定路径，
     打包产物里只允许出现在代码/二进制中，不允许出现在配置里）

信息级（不拦截）：构建机路径（仓库根目录等）会出现在 PyInstaller 产物里，
这是正常现象，只罗列不判失败。

退出码：0 = 通过；1 = 有红线命中。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent.parent / "dist-bundle"

# 本机用户名（红线按它动态生成，换机器不用改脚本）
import os  # noqa: E402
USERNAME = os.environ.get("USERNAME") or Path.home().name

# 个人模式：命中任一即失败（泄露即被人盗刷/定位到人）
# ⚠️ 通用部分写死；**本机专属的部分**（用户名、模型目录名、显卡型号…）
#    从 .githooks/local-patterns.txt 读（那个文件不进版本库），
#    这样这个脚本本身可以公开，谁都能用来验自己的整合包。
RED_LINES = ["tvly-",
             "C:" + chr(92) + "Users" + chr(92) + USERNAME]

# 本机专属模式（可选文件，不进版本库）
LOCAL_PAT = Path(__file__).resolve().parent.parent / ".githooks" / "local-patterns.txt"
if LOCAL_PAT.is_file():
    for line in LOCAL_PAT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            RED_LINES.append(line)

# 信息级：PyInstaller 包的 _internal/*.pyd 的 PE 调试目录里带上游
# python-build-standalone 构建机的 PDB 路径（某个构建账号的家目录），
# 任何 CPython Windows 构建都有、不含本机用户信息 —— 只罗列不拦截。
INFO_PAT = "C:" + chr(92) + "Users" + chr(92)
CLEAN_KEYS_EMPTY = ("tavily_key", "llama_server_exe", "ollama_dir",
                    "import_dir", "last_model")
CLEAN_KEYS_DICT = ("imported_models", "model_units", "model_variants",
                   "mmproj_models", "draft_models", "ctx_measured",
                   "ctx_measured_env", "ctx_measured_note",
                   "attach_overrides", "moe_offload", "fit_target",
                   "model_ctx_overrides")


def main() -> int:
    if not BUNDLE.is_dir():
        print(f"❌ 没有 dist-bundle/：{BUNDLE}")
        return 1

    files = [f for f in BUNDLE.rglob("*") if f.is_file()]
    print(f"扫描 {len(files)} 个文件（{BUNDLE}）…")

    hits: list[tuple[str, str]] = []
    info_hits: list[str] = []
    total = 0
    for f in files:
        try:
            data = f.read_bytes()
        except OSError as e:
            print(f"  ⚠ 读不了 {f}: {e}")
            continue
        total += len(data)
        for pat in RED_LINES:
            if pat.encode("utf-8") in data:
                hits.append((pat, str(f.relative_to(BUNDLE))))
        if INFO_PAT.encode("utf-8") in data:
            info_hits.append(str(f.relative_to(BUNDLE)))

    print(f"总大小 {total / 1024 ** 3:.2f} GB\n")

    print(f"== 红线扫描（{len(RED_LINES)} 个个人模式，必须零命中）==")
    if hits:
        for pat, f in hits:
            print(f"  ❌ {pat!r} 命中：{f}")
    else:
        print("  ✅ 个人模式全部零命中")

    print("\n== data/config.json 专项 ==")
    cfgp = BUNDLE / "data" / "config.json"
    ok_cfg = True
    if not cfgp.is_file():
        print("  ❌ 没有 data/config.json")
        ok_cfg = False
    else:
        raw = cfgp.read_bytes()
        if b".kaiwu" in raw:
            print("  ❌ data/config.json 含 .kaiwu")
            ok_cfg = False
        cfg = json.loads(raw.decode("utf-8"))
        for k in CLEAN_KEYS_EMPTY:
            if cfg.get(k):
                print(f"  ❌ {k} 非空：{cfg[k]!r}")
                ok_cfg = False
        for k in CLEAN_KEYS_DICT:
            if cfg.get(k):
                print(f"  ❌ {k} 非空（{len(cfg[k])} 项）")
                ok_cfg = False
        if ok_cfg:
            print("  ✅ 本机键全空，无 .kaiwu")

    print("\n== 信息级：上游构建机路径（PyInstaller 正常现象，不拦截）==")
    if info_hits:
        print(f"  · {len(info_hits)} 个文件的 PE 调试目录含上游 PDB 路径"
              f"（python-build-standalone 构建机，非本机信息）")
        for f in info_hits[:3]:
            print(f"    · {f}")
        if len(info_hits) > 3:
            print(f"    · …另 {len(info_hits) - 3} 个")
    else:
        print("  · 无")

    passed = not hits and ok_cfg
    print(f"\nT5 隐私验收：{'✅ 通过' if passed else '❌ 未通过'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
