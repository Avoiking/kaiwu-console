"""模型单元扫描 / 登记 / 配对全链路测试（2026-10-15）。

⚠️ 全程使用临时目录 + 临时 config，**不碰仓库里的 data/config.json**。
跑法：python _test_model_scan.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import kaiwu_manager, model_scan, webui   # noqa: E402

PY = sys.executable
FAILED = []


def check(cond, msg):
    tag = "ok " if cond else "FAIL"
    print(f"[{tag}] {msg}")
    if not cond:
        FAILED.append(msg)


def touch(p: Path, size=1024):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * size)


# ------------------------------------------------------------ fixture
TMP = Path(tempfile.mkdtemp(prefix="kaiwu_scan_"))
NEW = TMP / "models_new"          # 新布局：一模型一文件夹
LEG = TMP / "models_legacy"       # 旧布局：平铺

# ① gemma：主权重 + 视觉塔（名字只写到 gemma-4-26B）+ MTP 头
touch(NEW / "gemma-4-26B-A4B-it-ud" / "gemma-4-26B-A4B-it-ud-IQ3_S.gguf", 11_000)
touch(NEW / "gemma-4-26B-A4B-it-ud" / "mmproj-gemma-4-26B-BF16.gguf", 1_160)
touch(NEW / "gemma-4-26B-A4B-it-ud" / "mtp-gemma-4-26B-A4B-it-Q8_0.gguf", 460)
# ② Qwen3.5-4B：两个量化变体（Q5 更大 = 默认激活）+ 视觉塔
touch(NEW / "Qwen3.5-4B-UD" / "Qwen3.5-4B-UD-Q5_K_XL.gguf", 3_000)
touch(NEW / "Qwen3.5-4B-UD" / "Qwen3.5-4B-UD-Q4_K_M.gguf", 2_400)
touch(NEW / "Qwen3.5-4B-UD" / "mmproj-Qwen3.5-4B-F16.gguf", 650)
# ③ 两个单权重模型
touch(NEW / "Qwen3.5-9B-UD" / "Qwen3.5-9B-UD-Q4_K_XL.gguf", 5_700)
touch(NEW / "Ornith-1.5-9B" / "Ornith-1.5-9B-Q4_K_M.gguf", 5_400)
touch(NEW / "josie-4b-instruct-32k" / "josie-4b-instruct-32k.gguf", 2_300)
# ④ 一个「文件夹私有但塔名不带底座名」的模型 → 应按 folder 规则挂上
touch(NEW / "mystery-7b" / "mystery-7b-Q4_K_M.gguf", 4_000)
touch(NEW / "mystery-7b" / "mmproj-BF16.gguf", 900)
# ⑤ 旧布局：平铺 + 认不出归属的塔（mmproj-BF16）+ 一个 MTP 头
touch(LEG / "model-a-Q4_K_M.gguf", 2_000)
touch(LEG / "model-b-Q4_K_M.gguf", 2_100)
touch(LEG / "mmproj-BF16.gguf", 800)
touch(LEG / "mtp-something-Q8_0.gguf", 300)

# ------------------------------------------------------------ 临时 config
CFG = TMP / "config.json"


def fake_load():
    if CFG.exists():
        return json.loads(CFG.read_text(encoding="utf-8"))
    return {}


def fake_save(cfg):
    CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


webui.load_config = fake_load
webui.save_config = fake_save
kaiwu_manager.load_config = fake_load


class _MgrStub:
    def __init__(self):
        self._st = ("stopped", "")
        self.stopped = False

    def state(self):
        return self._st

    def stop(self, cfg):
        self.stopped = True


class _UIStub:
    def __init__(self):
        self.manager = _MgrStub()


UI = _UIStub()

# ============================================================ 1. scan_units
print("\n== scan_units ==")
units = model_scan.scan_units(NEW)
folders = sorted(u.folder.name for u in units)
check(folders == ["Ornith-1.5-9B", "Qwen3.5-4B-UD", "Qwen3.5-9B-UD",
                  "gemma-4-26B-A4B-it-ud", "josie-4b-instruct-32k",
                  "mystery-7b"], f"6 个模型单元：{folders}")
gemma = next(u for u in units if u.folder.name.startswith("gemma"))
q4 = next(u for u in units if u.folder.name == "Qwen3.5-4B-UD")
check(list(gemma.groups) == ["gemma-4-26B-A4B-it-ud"], "gemma 一组一主权重")
check(sorted(q4.groups) == ["Qwen3.5-4B-UD"] and len(q4.groups["Qwen3.5-4B-UD"]) == 2,
      "Qwen3.5-4B 两个量化变体归一组")
check(len(gemma.towers) == 1 and len(gemma.drafts) == 1, "gemma 塔/头各一")

# ============================================================ 2. bind_sidecars
print("\n== bind_sidecars ==")
b = model_scan.bind_sidecars(gemma, "gemma-4-26B-A4B-it-ud")
v, vhow = b["vision"]
d, dhow = b["draft"]
check(v and v.name == "mmproj-gemma-4-26B-BF16.gguf" and vhow == "name",
      f"gemma 视觉塔按文件名自证（{vhow}）")
check(d and d.name.startswith("mtp-gemma") and dhow == "name",
      f"gemma MTP 头按文件名自证（{dhow}）")
b = model_scan.bind_sidecars(q4, "Qwen3.5-4B-UD")
check(b["vision"][0].name == "mmproj-Qwen3.5-4B-F16.gguf", "q4 视觉塔自证")
mystery = next(u for u in units if u.folder.name == "mystery-7b")
b = model_scan.bind_sidecars(mystery, "mystery-7b")
check(b["vision"][0].name == "mmproj-BF16.gguf" and b["vision"][1] == "folder",
      f"塔名不带底座名 → 文件夹私有规则挂上（{b['vision'][1]}）")
leg_units = model_scan.scan_units(LEG)
leg = leg_units[0]
check(not leg.private, "旧布局目录不私有（2 个模型）")
b = model_scan.bind_sidecars(leg, "model-a")
check(b["vision"][0] is None, "旧布局多模型混放 → 塔认不出归属，不猜")

# ============================================================ 3. models_add
print("\n== models_add（新布局） ==")
r = webui.WebUI.models_add(UI, str(NEW))
check(r["ok"] and len(r["added"]) == 6, f"登记 6 个模型：{r['added']}")
cfg = fake_load()
imp = cfg["imported_models"]
check(set(imp) == {"gemma-4-26B-A4B-it-ud", "Qwen3.5-4B-UD", "Qwen3.5-9B-UD",
                   "Ornith-1.5-9B", "josie-4b-instruct-32k", "mystery-7b"},
      f"imported_models 键 = 底座名：{sorted(imp)}")
check(imp["Qwen3.5-4B-UD"].endswith("Q5_K_XL.gguf"), "默认激活最大（质量最高）的变体")
check(set(cfg["model_variants"]["Qwen3.5-4B-UD"]) == {"Q5_K_XL", "Q4_K_M"},
      "model_variants 两个量化档")
check("model_variants" not in cfg or "mystery-7b" not in cfg["model_variants"],
      "单变体不写 model_variants")
check(len(cfg["model_units"]) == 6, "model_units 登记 6 个文件夹")
check(cfg["mmproj_models"]["gemma-4-26B-A4B-it-ud"].endswith("mmproj-gemma-4-26B-BF16.gguf"),
      "gemma 塔已绑")
check(cfg["mmproj_models"]["mystery-7b"].endswith("mmproj-BF16.gguf"), "mystery 塔已绑（folder 规则）")
check(cfg["draft_models"]["gemma-4-26B-A4B-it-ud"]["path"].endswith("mtp-gemma-4-26B-A4B-it-Q8_0.gguf"),
      "gemma MTP 头已绑")
check(all("model-a" not in k for k in cfg["mmproj_models"]), "不存在的模型不乱绑")

print("\n== models_add（旧布局，兼容性） ==")
r2 = webui.WebUI.models_add(UI, str(LEG))
check(r2["ok"] and set(r2["added"]) == {"model-a-Q4_K_M", "model-b-Q4_K_M"},
      f"旧布局逐个登记、完整文件名：{r2['added']}")
cfg = fake_load()
check(not any(k.startswith(("model-a", "model-b")) for k in cfg.get("model_variants") or {}),
      "旧布局不写量化变体")
check(not any("model-a" in k or "model-b" in k for k in cfg["mmproj_models"]),
      "旧布局塔认不出归属 → 不自动配")

print("\n== models_add（幂等） ==")
r3 = webui.WebUI.models_add(UI, str(NEW))
check(r3["ok"] and not r3["added"] and len(r3["skipped"]) == 6,
      f"重复添加全部跳过：{r3['skipped']}")

# ============================================================ 4. list_models
print("\n== list_models ==")
mgr = kaiwu_manager.KaiwuManager()
models = {m["name"]: m for m in mgr.list_models(cfg)}
check(len(models) == 8, f"8 个模型（6 新 + 2 旧）：{sorted(models)}")
q4m = models["Qwen3.5-4B-UD"]
check(set(q4m["variants"]) == {"Q5_K_XL", "Q4_K_M"} and q4m["active"] == "Q5_K_XL",
      f"变体表 + 激活项：{q4m['variants']} / {q4m['active']}")
check(models["mystery-7b"]["variants"].keys() == {"Q4_K_M"}, "单变体也进表（供下拉）")
check(models["model-a-Q4_K_M"]["variants"].keys() == {"Q4_K_M"}, "旧布局条目无多余变体")

# ============================================================ 5. set_variant
print("\n== set_variant ==")
r = webui.WebUI.set_variant(UI, "Qwen3.5-4B-UD", "Q4_K_M")
check(r["ok"] and fake_load()["imported_models"]["Qwen3.5-4B-UD"].endswith("Q4_K_M.gguf"),
      "切换激活变体写入 imported_models")
r = webui.WebUI.set_variant(UI, "Qwen3.5-4B-UD", "Q9_K_XL")
check(not r["ok"], "不存在的量化档被拒绝")

print("\n== set_variant（新发现：刚扔进文件夹还没登记的量化档） ==")
touch(NEW / "Qwen3.5-4B-UD" / "Qwen3.5-4B-UD-Q3_K_M.gguf", 2_000)
m2 = {m["name"]: m for m in mgr.list_models(fake_load())}
check("Q3_K_M" in m2["Qwen3.5-4B-UD"]["variants"],
      f"新文件自动出现在卡片下拉：{list(m2['Qwen3.5-4B-UD']['variants'])}")
r = webui.WebUI.set_variant(UI, "Qwen3.5-4B-UD", "Q3_K_M")
check(r["ok"] and fake_load()["imported_models"]["Qwen3.5-4B-UD"].endswith("Q3_K_M.gguf"),
      "选新量化档成功")
check("Q3_K_M" in fake_load()["model_variants"]["Qwen3.5-4B-UD"],
      "新量化档补登进 model_variants（下次不用扫目录）")

# ============================================================ 6. sidecar_pick
print("\n== sidecar_pick ==")
main_w = str(NEW / "mystery-7b" / "mystery-7b-Q4_K_M.gguf")
r = webui.WebUI.sidecar_pick(UI, "mystery-7b", "vision", main_w)
check(not r["ok"], "主权重不能当视觉塔挂")
r = webui.WebUI.sidecar_pick(UI, "mystery-7b", "vision",
                             str(NEW / "mystery-7b" / "mmproj-BF16.gguf"))
check(r["ok"] and fake_load()["mmproj_models"]["mystery-7b"].endswith("mmproj-BF16.gguf"),
      "手动指定视觉塔写入显式映射")

# ============================================================ 7. 解析来源 + sole_owner
print("\n== _resolve_*_ex / _sole_owner_of ==")
cfg2 = fake_load()
mp = imp_path = cfg2["imported_models"]["gemma-4-26B-A4B-it-ud"]
p, how = kaiwu_manager.KaiwuManager._resolve_mmproj_ex(cfg2, "gemma-4-26B-A4B-it-ud", mp)
check(how == "config", f"显式映射来源 = config（{how}）")
# 清掉显式映射，走加载时兜底：应分别按 name / folder 认出
cfg3 = {k: v for k, v in cfg2.items() if k not in ("mmproj_models", "draft_models")}
p, how = kaiwu_manager.KaiwuManager._resolve_mmproj_ex(cfg3, "gemma-4-26B-A4B-it-ud", mp)
check(p.endswith("mmproj-gemma-4-26B-BF16.gguf") and how == "name",
      f"无显式映射时按文件名认出（{how}）")
mp2 = cfg3["imported_models"]["mystery-7b"]
p, how = kaiwu_manager.KaiwuManager._resolve_mmproj_ex(cfg3, "mystery-7b", mp2)
check(p.endswith("mmproj-BF16.gguf") and how == "folder",
      f"塔名不带底座名 → 文件夹私有认出（{how}）")
mpl = cfg3["imported_models"]["model-a-Q4_K_M"]
p, how = kaiwu_manager.KaiwuManager._resolve_mmproj_ex(cfg3, "model-a-Q4_K_M", mpl)
check(p == "" and how == "", "旧布局认不出 → 不挂（宁缺勿错）")
check(kaiwu_manager.KaiwuManager._sole_owner_of(mp) is True, "gemma 目录 = 私有")
check(kaiwu_manager.KaiwuManager._sole_owner_of(
    cfg3["imported_models"]["Qwen3.5-4B-UD"]) is True, "多变体文件夹仍算私有")
check(kaiwu_manager.KaiwuManager._sole_owner_of(mpl) is False, "旧布局 = 共享目录")

# ============================================================ 8. pending_sidecars
print("\n== pending_sidecars ==")
pend = kaiwu_manager.KaiwuManager.pending_sidecars(mpl, "vision")
check([x["name"] for x in pend] == ["mmproj-BF16.gguf"], f"待选择列表：{pend}")

shutil.rmtree(TMP, ignore_errors=True)
print("\n" + ("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}"))
sys.exit(1 if FAILED else 0)
