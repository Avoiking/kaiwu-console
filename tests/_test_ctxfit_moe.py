"""MoE 混合卸载测量的规格测试（2026-10-17，真机日志作夹具，不开推理服务）。

背景（用户投诉 + 真机复测）：gemma-4-26B-A4B-it-UD 是 MoE（128 专家、
每 token 激活 8 个），15.1G 权重 vs 8G 显存。2026-10-16 的修复把"权重
超出显存"直接判成"装不下、速度约 2~3 tok/s（编造）、换更大显卡"——
完全错误。真机复测（8G 独显，b10951，完整配置含 MTP 草稿
+ 视觉塔 + KV q8_0）：

    --cpu-moe @32768  → 19.5 tok/s，CUDA0 model buffer 只 1964 MiB
    --cpu-moe @98304  → 18.7 tok/s
    --cpu-moe @131072 → 16.3 tok/s（VRAM 7172/8151）
    默认卸载 @98304   → 12.2 tok/s（fit 自动把权重切去 CPU_Mapped）
    --cpu-moe @196608（挂草稿）→ 1.18 tok/s，VRAM 贴顶 —— 层数 31/31
        全在 GPU，层数判据完全看不出来，只有实测 tok/s 能抓住

即：MoE 的权重在内存 mmap 是**正常形态**而非病态；判据必须是"层数在
GPU + 实测速度达门槛"，窗口由真实测量定，备注写实测数不写编造数。

本测试锁住的行为：
  1. 日志 print_info: n_expert > 0 → 认出 MoE（含挂了草稿/塔的主模型段锚定）
  2. 稠密日志（Ornith）不误判成 MoE
  2b. _detect_moe 三级判据：引擎自报为权威、名称兜底仅引擎没开口时用、
      没有信号保守按稠密 —— 以后加新模型不用改代码
  3. _estimate_ctx(moe=True) 对混合账目给出真实估值（不再 best=0 放弃）；
     同账目 moe=False 仍按稠密放弃（防回归）
  4. 三个备注构造函数：只写实测数、不写编造速度、MoE 好形态用 ℹ 不说教；
     稠密 / MoE 门槛分离（ctx_dense_min_tps=30 / ctx_moe_min_tps=15）
  5. _moe_offload_mode / _llama_supports / _measure_tps 的契约
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.kaiwu_manager as _km
from app.kaiwu_manager import KaiwuManager

PASS = FAIL = 0


def ok(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}")


# ---- gemma-4-26B-A4B-IQ3_S 默认卸载真实日志（2026-10-17 捕获，无伴生文件）----
GEMMA_DEFAULT = [
    "0.00.727.745 I srv    load_model: loading model 'D:/test-models/gemma-4-26B-A4B-it-UD/gemma-4-26B-A4B-it-UD-IQ3_S.gguf'",
    "0.08.014.386 I print_info: n_ctx_train           = 262144",
    "0.08.014.387 I print_info: n_embd                = 2816",
    "0.08.014.388 I print_info: n_layer               = 30",
    "0.08.014.437 I print_info: n_expert              = 128",
    "0.08.014.437 I print_info: n_expert_used         = 8",
    "0.08.014.437 I print_info: n_expert_groups       = 0",
    "0.08.014.496 I load_tensors: loading model tensors, this can take a while... (load_mode = mmap)",
    "0.09.445.972 I load_tensors: offloaded 31/31 layers to GPU",
    "0.09.445.976 I load_tensors:   CPU_Mapped model buffer size = 10745.48 MiB",
    "0.09.445.978 I load_tensors:        CUDA0 model buffer size =  4733.41 MiB",
    "0.14.120.432 I llama_context: n_ctx                 = 32768",
    "0.14.166.840 I llama_kv_cache: size =  340.00 MiB ( 32768 cells,   5 layers,  4/1 seqs), K (q8_0):  170.00 MiB, V (q8_0):  170.00 MiB",
    "0.14.230.823 I llama_kv_cache: size =  478.12 MiB (  4608 cells,  25 layers,  4/1 seqs), K (q8_0):  239.06 MiB, V (q8_0):  239.06 MiB",
    "0.14.249.544 I sched_reserve:      CUDA0 compute buffer size =   262.21 MiB",
    "0.15.574.889 I srv  llama_server: model loaded",
]

# ---- gemma --cpu-moe 真实日志（专家全走内存，显存只驻注意力/嵌入）----
GEMMA_CPU_MOE = [
    "0.00.196.433 I srv    load_model: loading model 'D:/test-models/gemma-4-26B-A4B-it-UD/gemma-4-26B-A4B-it-UD-IQ3_S.gguf'",
    "0.01.000.000 I print_info: n_ctx_train           = 262144",
    "0.01.001.000 I print_info: n_layer               = 30",
    "0.01.002.000 I print_info: n_expert              = 128",
    "0.01.002.000 I print_info: n_expert_used         = 8",
    "0.02.716.997 I load_tensors: offloaded 31/31 layers to GPU",
    "0.02.717.001 I load_tensors:   CPU_Mapped model buffer size = 10745.48 MiB",
    "0.02.717.001 I load_tensors:        CUDA0 model buffer size =  1963.99 MiB",
    "0.03.502.300 I llama_context: n_ctx                 = 32768",
    "0.03.504.297 I llama_kv_cache: size =  340.00 MiB ( 32768 cells,   5 layers,  4/1 seqs), K (q8_0):  272.00 MiB, V (q8_0):  272.00 MiB",
    "0.03.516.979 I llama_kv_cache: size =  478.12 MiB (  4608 cells,  25 layers,  4/1 seqs), K (q8_0):  239.06 MiB, V (q8_0):  239.06 MiB",
    "0.03.543.431 I sched_reserve:      CUDA0 compute buffer size =   357.93 MiB",
    "0.04.000.000 I srv  llama_server: model loaded",
]

# ---- 完整配置（挂了 MTP 草稿）——草稿段是稠密的 n_expert=0，
#      主模型段锚定必须仍认出 MoE、仍取主模型的账目（不是草稿的 425 MiB）----
GEMMA_WITH_DRAFT = GEMMA_CPU_MOE + [
    "0.10.000.000 I srv    load_model: loading draft model 'D:/test-models/gemma-4-26B-A4B-it-UD/mtp-gemma-4-26B-A4B-it-Q8_0.gguf'",
    "0.10.001.000 I print_info: n_expert              = 0",
    "0.10.002.000 I print_info: n_expert_used         = 0",
    "0.10.003.000 I load_tensors: offloaded 5/5 layers to GPU",
    "0.10.004.000 I load_tensors:   CPU_Mapped model buffer size =   272.00 MiB",
    "0.10.005.000 I load_tensors:        CUDA0 model buffer size =   425.34 MiB",
    "0.10.006.000 I sched_reserve:      CUDA0 compute buffer size =   528.78 MiB",
]

# ---- 稠密模型对照（Ornith 真实日志，见 _test_ctxfit_ornith.py）----
ORNITH = [
    "0.00.172.959 I srv    load_model: loading model 'D:\\models\\Ornith-1.5-9B\\Ornith-1.5-9B-Q4_K_M.gguf'",
    "0.01.640.146 I print_info: n_ctx_train           = 262144",
    "0.01.640.147 I print_info: n_layer               = 32",
    "0.01.640.148 I print_info: n_layer_all           = 33",
    "0.02.018.883 I load_tensors: offloaded 33/34 layers to GPU",
    "0.02.018.886 I load_tensors:   CPU_Mapped model buffer size =   683.55 MiB",
    "0.02.018.888 I load_tensors:        CUDA0 model buffer size =  4674.33 MiB",
    "0.03.502.300 I llama_context: n_ctx                 = 32768",
    "0.03.504.297 I llama_kv_cache: size =  544.00 MiB ( 32768 cells,   8 layers,  4/1 seqs), K (q8_0):  272.00 MiB, V (q8_0):  272.00 MiB",
    "0.03.543.431 I sched_reserve:      CUDA0 compute buffer size =   239.66 MiB",
]

CFG = {
    "ctx_vram_reserve_mib": 512,
    "ctx_estimate_ratio": 0.42,
    "ctx_moe_min_tps": 15.0,
    "ctx_dense_min_tps": 30.0,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "ctx_tps_rounds": 3,
    "ctx_tps_tokens": 48,
    "ctx_tps_timeout_sec": 120,
}

# 账目解析只依赖日志缓冲；估值要 nvidia-smi 数字 —— 钉死本机实测值，
# 免得测试在别人的机器上漂移（8151 MiB = 8G 独显）。
KaiwuManager.total_vram_mib = staticmethod(lambda: 8151)

# [5] 段会把 _ctx_cache_key 打桩成"恒定 @IQ3_S"（那里只想验 moe_offload 的
# 键来源），打桩后不还原 —— [10] 段要验真实现（按权重路径算量化档），先留一份。
_REAL_CTX_KEY = KaiwuManager._ctx_cache_key


def mgr_with(lines):
    m = KaiwuManager()
    m._logbuf.clear()
    m._logbuf.extend(lines)
    return m


print("\n[1] MoE 身份识别（日志 print_info n_expert）")
acct_g = mgr_with(GEMMA_DEFAULT)._parse_vram_log()
ok(acct_g is not None, "gemma 账目解析成功")
ok(acct_g.get("n_expert") == 128, f"n_expert=128（实际 {acct_g.get('n_expert')}）")
ok(acct_g.get("n_expert_used") == 8, f"n_expert_used=8（实际 {acct_g.get('n_expert_used')}）")
ok(KaiwuManager._is_moe(acct_g) is True, "gemma 认出是 MoE")
acct_o = mgr_with(ORNITH)._parse_vram_log()
ok(KaiwuManager._is_moe(acct_o) is False, "Ornith（日志无 n_expert）不误判成 MoE")
ok(KaiwuManager._is_moe(None) is False, "空账目不误判")

print("\n[1b] _detect_moe 三级判据（对任何新模型成立，不靠白名单）")
ok(KaiwuManager._detect_moe(acct_g, "随便什么新名字") is True,
   "引擎自报 n_expert=128 → MoE（与名字无关，新模型免改代码）")
ok(acct_o.get("print_info_seen") is True,
   "Ornith 账目记录 print_info_seen（否定信号可信的前提）")
ok(KaiwuManager._detect_moe(acct_o, "Ornith-1.5-9B") is False,
   "print_info 段在且无 n_expert → 引擎明确说稠密")
acct_silent = {k: v for k, v in acct_o.items() if k != "print_info_seen"}
ok(KaiwuManager._detect_moe(acct_silent, "Mixtral-8x7B") is True,
   "引擎没开口（未来改字段名）+ mixtral → 名称兜底认 MoE")
ok(KaiwuManager._detect_moe(acct_silent, "gemma-4-26B-A4B") is True,
   "引擎没开口 + -A4B 激活参数写法 → 名称兜底认 MoE")
ok(KaiwuManager._detect_moe(acct_silent, "Ornith-1.5-9B") is False,
   "引擎没开口 + 名称无标记 → 保守按稠密（漏判只是窗口保守+备注如实）")
ok(KaiwuManager._detect_moe(None, "Qwen3-30B-A3B") is True,
   "完全没账目 + A3B 写法 → 名称兜底认 MoE")
for _nm, _want in [("Qwen3.5-35B-A3B", True), ("qwen3-moe", True),
                   ("GLM-4.5", False), ("MathExpert-7B", False),
                   ("Ornith-1.5-9B", False), ("some_dense_model", False)]:
    ok(KaiwuManager._looks_moe_by_name(_nm) is _want,
       f"名称标记 {_nm!r} → {_want}（高精度优先，不误伤稠密微调名）")

print("\n[2] 主模型段锚定：挂了草稿（稠密 n_expert=0）仍认 MoE、仍取主账目")
acct_d = mgr_with(GEMMA_WITH_DRAFT)._parse_vram_log()
ok(KaiwuManager._is_moe(acct_d) is True,
   f"草稿段 n_expert=0 不污染主模型判定（实际 {acct_d.get('n_expert')}）")
ok(abs(acct_d.get("model_mib", 0) - 1963.99) < 0.01,
   f"model_mib 仍是主模型的 1963.99，不是草稿的 425.34（实际 {acct_d.get('model_mib')}）")

print("\n[3] MoE 混合账目的估值（moe=True 不再放弃；moe=False 仍按稠密放弃）")
m = mgr_with(GEMMA_CPU_MOE)
est_moe = m._estimate_ctx(CFG, mem=acct_d, moe=True)
# (8151-1963.99-357.93-512-478.12)/0.010376×0.42 向下取 4096 整数倍。
# 192512（188K）比本机实测成立的 196608 低一档 —— 校准系数故意保守，
# 真正定夺那档的是后面的实测加载 + 测速，不是这个数本身。
ok(est_moe is not None and est_moe.get("best") == 192512,
   f"cpu-moe 账目 → 192512（不再 best=0 放弃；实际 "
   f"{est_moe and est_moe.get('best')}）")
ok(not est_moe.get("reason"), "MoE 模式不写'算式不适用'的放弃理由")
est_dense = m._estimate_ctx(CFG, mem=acct_d, moe=False)
ok(est_dense is not None and est_dense.get("best") == 0
   and "无法全驻" in (est_dense.get("reason") or ""),
   "同账目 moe=False → best=0 + 放弃理由（稠密行为防回归）")
est_def = mgr_with(GEMMA_DEFAULT)._estimate_ctx(CFG, mem=acct_g, moe=True)
ok(est_def is not None and est_def.get("best") > 32768,
   f"默认卸载账目（model 4733）也给出 >32768 的估值（实际 "
   f"{est_def and est_def.get('best')}）")

print("\n[4] 备注构造函数：只写实测数，不写编造速度，不说教")
n1 = KaiwuManager._note_moe_hybrid(acct_d, 18.7, 98304)
ok(n1.startswith("ℹ"), "MoE 好形态用 ℹ（信息）而非 ⚠（警告）")
ok("18.7 tok/s" in n1 and "96K" in n1 and "128" in n1,
   f"备注带实测速度/窗口/专家数（实际：{n1}）")
ok("换更小量化" not in n1 and "才能全速" not in n1,
   "不再写'换更小量化档/更大显卡'的投降式建议")
ok("2~3" not in n1, "没有编造的速度区间")
n2 = KaiwuManager._note_moe_degraded(acct_d, 1.2, 32768, 5.0)
ok(n2.startswith("⚠") and "1.2 tok/s" in n2 and "5 tok/s" in n2,
   f"摸不到门槛的 MoE：实测 1.2 vs 门槛 5（实际：{n2}）")
ok("加内存" not in n2 and "更大显存卡" not in n2,
   "MoE 降级备注也不劝升级硬件（用户守则：只给榨干硬件方向的建议）")
n3 = KaiwuManager._note_dense_degraded(acct_g, 1.8, 32768, 8151)
ok("1.8 tok/s" in n3 and "15.1G" in n3 and "8G" in n3,
   f"稠密降级备注写实测 1.8（实际：{n3}）")
ok("2~3" not in n3, "稠密备注同样没有编造的'2~3 tok/s'")
n4 = KaiwuManager._note_dense_degraded(acct_g, None, 32768, 8151)
ok("未测到" in n4, "测不到速度时如实说'未测到'，不猜")

# Ornith-1.5-9B Q5_K_XL 真机（2026-10-17）：31/34 层、959 MiB/5.2G=18% 走
# 内存 mmap，旧逻辑按 10% 比例阈值判它病态并劝人换显卡 —— 可它实测 31.5 tok/s。
acct_o = {"model_mib": 4366.0, "cpu_mapped_mib": 959.39,
          "on_gpu": 31, "total_layers": 34}
ok(KaiwuManager._weights_fully_resident(acct_o) is False,
   "Ornith 这档确实过不了 residency（越过 10% 比例阈值）")
n5 = KaiwuManager._note_dense_partial(acct_o, 31.5, 32768, 5.0)
ok(n5.startswith("ℹ"), "稠密混合形态（部分权重在内存但够快）用 ℹ 而非 ⚠")
ok("31.5 tok/s" in n5 and "18%" in n5 and "4.3G" in n5,
   f"ℹ 备注写实测速度/占比/显存驻留（实际：{n5}）")
ok("换更小量化" not in n5 and "更大显存卡" not in n5, "ℹ 备注不劝换硬件")
n6 = KaiwuManager._note_dense_degraded(acct_o, 31.5, 32768, 8151, 5.0)
ok("超出显存" not in n6, "权重 5.2G 装得进 8G → 不许写'超出显存'（旧文案在说谎）")
ok("没能全驻" in n6, f"装得下时如实说'没能全驻'（实际：{n6}）")
ok("换更小量化" not in n6, "实测 31.5 tok/s 时不许劝换显卡")
ok("31.5 tok/s" in n6, "降级备注仍带实测速度")
n7 = KaiwuManager._note_dense_degraded(acct_g, 1.2, 32768, 8151, 5.0)
ok("超出显存" in n7 and "换更小量化" in n7,
   "真的装不下且真的慢 → 才写'超出显存'并给建议")

print("\n[4b] 稠密 / MoE 门槛分离（2026-10-17 用户拍板：小稠密至少 30）")
from app.config import DEFAULTS
ok(DEFAULTS.get("ctx_dense_min_tps") == 30.0,
   f"DEFAULTS 带 ctx_dense_min_tps=30.0（实际 {DEFAULTS.get('ctx_dense_min_tps')}）")
ok(DEFAULTS.get("ctx_moe_min_tps") == 15.0,
   "MoE 门槛 15（弱机器上 gemma 级 MoE 稳定 15~18；8.2 悬崖档照样拦）")
n8 = KaiwuManager._note_dense_partial(acct_o, 35.0, 61440, 30.0)
ok("判定门槛 30 tok/s" in n8,
   f"稠密混合备注写实际生效的稠密门槛 30，不是 MoE 的 5（实际：{n8}）")
ok("35.0 tok/s" in n8 and "60K" in n8 and "18%" in n8,
   "备注仍带实测速度 / 窗口 / 占比")
n9 = KaiwuManager._note_dense_degraded(acct_o, 12.4, 114688, 8151, 30.0)
ok("12.4 tok/s" in n9 and "30 tok/s" in n9,
   f"降级备注同时带实测速度和稠密门槛（实际：{n9}）")

print("\n[5] _moe_offload_mode：按模型@量化档读混合卸载记录")
cfg_m = {"moe_offload": {"gemma-4-26B-A4B-it-UD@IQ3_S": "cpu"}}
KaiwuManager._ctx_cache_key = staticmethod(
    lambda cfg, model, path="": f"{model}@IQ3_S")
ok(KaiwuManager._moe_offload_mode(cfg_m, "gemma-4-26B-A4B-it-UD") == "cpu",
   "记了 cpu 的档位返回 cpu（加载时会带 --cpu-moe）")
ok(KaiwuManager._moe_offload_mode({}, "gemma-4-26B-A4B-it-UD") == "",
   "没记录的模型返回空（不加旗标）")
ok(KaiwuManager._moe_offload_mode(cfg_m, "Ornith-1.5-9B") == "",
   "别的模型不受影响")

print("\n[6] _llama_supports：二进制能力探测（缓存 + 缺失兜底）")
m2 = mgr_with([])
_orig_resolve = _km.resolve_llama_server
try:
    _km.resolve_llama_server = lambda cfg: ""
    ok(m2._llama_supports({"llama_server_exe": ""}, "--cpu-moe") is False,
       "找不到二进制 → False（调用方走默认卸载，不硬传旗标）")
    _km.resolve_llama_server = lambda cfg: "fake.exe"
    _km._llama_help_cache["fake.exe"] = "--cpu-moe keep all MoE weights in the CPU"
    ok(m2._llama_supports({"llama_server_exe": "fake.exe"}, "--cpu-moe") is True,
       "help 里有该旗标 → True")
    ok(m2._llama_supports({"llama_server_exe": "fake.exe"}, "--no-such-flag") is False,
       "help 里没有 → False")
    ok(m2._llama_supports({"llama_server_exe": "fake.exe"}, "--cpu-moe") is True,
       "二次调用走缓存（不重复起子进程）")
finally:
    _km.resolve_llama_server = _orig_resolve

print("\n[7] _measure_tps：真实测速的契约（读 timings，不猜）")


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


def _fake_post(payload, fail=False):
    def _post(url, json=None, timeout=None):
        if fail:
            raise OSError("connection reset")
        return _Resp(payload)
    return _post


_orig_post = _km.requests.post
try:
    _km.requests.post = _fake_post({"timings": {"predicted_per_second": 18.66}})
    ok(m2._measure_tps(CFG) == 18.66, "读 timings.predicted_per_second")
    _km.requests.post = _fake_post({"timings": {}})
    ok(m2._measure_tps(CFG) is None, "缺字段 → None（不猜）")
    _km.requests.post = _fake_post({}, fail=True)
    ok(m2._measure_tps(CFG) is None, "请求异常 → None（不猜）")
finally:
    _km.requests.post = _orig_post

print("\n[7b] _measure_tps 多次采样：丢预热、取其余最大值（单次采样冤案）")
# 背景：2026-10-17 自动测量在同一档位读到 2.04 tok/s 直接判死，几分钟后手动重
# 载同档位第一枪就 19.31。单次读数当唯一判据 = 抛硬币。所以现在要打几枪。
_measure_calls = []


def _fake_seq(tps_list, fail_at=None):
    """按顺序返回 timings；fail_at 里的下标抛异常；用完后沿用最后一个。"""
    it = iter(tps_list)

    def _post(url, json=None, timeout=None):
        idx = len(_measure_calls)
        _measure_calls.append(idx)
        if fail_at and idx in fail_at:
            raise OSError("connection reset")
        try:
            v = next(it)
        except StopIteration:
            v = tps_list[-1] if tps_list else None
        return _Resp({"timings": {"predicted_per_second": v}})
    return _post


_orig_post2 = _km.requests.post
try:
    # 预热那一枪明显偏低（真机 192512：首次 14.86，之后 22~27）
    _measure_calls.clear()
    _km.requests.post = _fake_seq([14.86, 22.31, 23.63])
    v = m2._measure_tps(CFG)
    ok(v == 23.63, f"丢掉预热那一枪，取后续最大（实际 {v}）")
    ok(len(m2._last_tps_samples) == 3, "原始采样留档（含预热枪，排查用）")
    ok(m2._last_tps_samples[0] == 14.86, "留档里第一枪是原始的低值，没被改写")

    # 偶发被别的进程拖累：只有一枪反映真实能力 —— 不能被拖累的那枪判死
    _measure_calls.clear()
    _km.requests.post = _fake_seq([19.0, 2.04, 24.0])
    ok(m2._measure_tps(CFG) == 24.0, "中间偶发掉速不拖累结论（取最大值）")

    # 真在换页的档位：每一枪都慢 → 再采样也救不回来，如实报慢
    _measure_calls.clear()
    _km.requests.post = _fake_seq([1.1, 1.2, 1.18])
    ok(m2._measure_tps(CFG) == 1.2, "每枪都慢 → 结论就是慢（不美化）")

    # 全挂 → None，绝不猜
    _measure_calls.clear()
    _km.requests.post = _fake_seq([None], fail_at={0, 1, 2})
    ok(m2._measure_tps(CFG) is None, "连打三次全失败 → None（不猜）")

    # 显式 rounds=1 → 兼容旧行为：就一枪，不丢
    _measure_calls.clear()
    _km.requests.post = _fake_seq([18.66, 30.0, 30.0])
    ok(m2._measure_tps(CFG, rounds=1) == 18.66, "rounds=1 时不丢那一枪（兼容旧行为）")
    ok(len(_measure_calls) == 1, "rounds=1 只打一次请求")

    # 部分请求失败：能测到的那一枪算数，不算"失败"
    _measure_calls.clear()
    _km.requests.post = _fake_seq([None, 21.0, None])
    ok(m2._measure_tps(CFG) == 21.0, "其中一枪失败不影响取其余的有效值")
finally:
    _km.requests.post = _orig_post2

print("\n[8] 环境指纹含伴生挂载状态（开关视觉塔/MTP → 旧窗口作废）")
import tempfile
_td = Path(tempfile.mkdtemp(prefix="moefp_"))
(_td / "M.gguf").write_bytes(b"x")
(_td / "mmproj-M.gguf").write_bytes(b"x")
(_td / "mtp-M.gguf").write_bytes(b"x")
CFG_ATT = {
    "cache_type_k": "q8_0", "cache_type_v": "q8_0",
    "attach_vision": True, "attach_draft": True,
    "imported_models": {"M": str(_td / "M.gguf")},
    "mmproj_models": {"M": str(_td / "mmproj-M.gguf")},
    "draft_models": {"M": {"path": str(_td / "mtp-M.gguf"),
                           "type": "draft-mtp", "n_max": 3}},
}
fp_on = m2._vram_fingerprint(CFG_ATT, "M")
ok(fp_on.get("attach") == "v1/d1", f"塔+头都挂 → v1/d1（实际 {fp_on.get('attach')}）")
cfg_off = dict(CFG_ATT, attach_overrides={"M": {"vision": False}})
fp_off = m2._vram_fingerprint(cfg_off, "M")
ok(fp_off.get("attach") == "v0/d1", f"关视觉 → v0/d1（实际 {fp_off.get('attach')}）")
_td2 = Path(tempfile.mkdtemp(prefix="moefp2_"))
(_td2 / "N.gguf").write_bytes(b"x")
cfg_noatt = {"cache_type_k": "q8_0", "cache_type_v": "q8_0",
             "attach_vision": True, "attach_draft": True,
             "imported_models": {"N": str(_td2 / "N.gguf")}}
ok(m2._vram_fingerprint(cfg_noatt, "N").get("attach") == "v0/d0",
   "目录里没塔/头 → v0/d0（挂了开关也不虚报）")
# 记录时的挂载状态 vs 当前挂载状态不一致 → 缓存失效（重测）
key_m = "M@Q4_K_M"
# 当前关了视觉（v0/d1），记录却是挂着塔时（v1/d1）→ 旧窗口作废
cfg_val = dict(cfg_off)
cfg_val["ctx_measured"] = {key_m: 40000}
cfg_val["ctx_measured_env"] = {key_m: fp_on}
ok(m2._ctx_cache_valid(cfg_val, key_m, "M") is False,
   "记录时挂着塔/头、现在关了 → 失效（旧窗口可能起不来）")
cfg_val2 = dict(cfg_off)
cfg_val2["ctx_measured"] = {key_m: 40000}
cfg_val2["ctx_measured_env"] = {key_m: fp_off}
ok(m2._ctx_cache_valid(cfg_val2, key_m, "M") is True,
   "挂载状态一致 → 缓存仍可信（不无谓重测）")
# 预填批量也并进指纹：改了它 compute buffer 涨 ~700 MiB，旧窗口不再成立
fp_b1 = m2._vram_fingerprint(dict(cfg_off), "M")     # 挂载 v0/d1、默认批量
fp_b2 = m2._vram_fingerprint(dict(cfg_off, prefill_batch=4096,
                                  prefill_ubatch=2048), "M")
ok(fp_b1.get("batch") != fp_b2.get("batch"),
   f"指纹含 batch 字段且随配置变（{fp_b1.get('batch')} vs {fp_b2.get('batch')}）")
cfg_b = dict(cfg_off)
cfg_b["ctx_measured"] = {key_m: 40000}
cfg_b["ctx_measured_env"] = {key_m: fp_b1}           # 记录与当前同为大默认批量
ok(m2._ctx_cache_valid(cfg_b, key_m, "M") is True, "同为默认批量 → 缓存有效")
cfg_b2 = dict(cfg_off, prefill_batch=4096, prefill_ubatch=2048)
cfg_b2["ctx_measured"] = {key_m: 40000}
cfg_b2["ctx_measured_env"] = {key_m: fp_b1}          # 记录时默认、现在大批量
ok(m2._ctx_cache_valid(cfg_b2, key_m, "M") is False,
   "开了大批量（compute buffer +700 MiB）→ 旧窗口作废重测")

print("\n[9] _prefill_args：预填批量旗标（实测 401→908 tok/s 的那组）")
ok(KaiwuManager._prefill_args({}) == [], "没配 → 空（用 llama.cpp 默认 2048/512）")
ok(KaiwuManager._prefill_args({"prefill_ubatch": 512}) == [],
   "ubatch=512 = 引擎默认 → 不加旗标")
ok(KaiwuManager._prefill_args({"prefill_batch": 4096, "prefill_ubatch": 2048})
   == ["-b", "4096", "-ub", "2048"], "默认配置 → -b 4096 -ub 2048")
ok(KaiwuManager._prefill_args({"prefill_ubatch": 2048})
   == ["-b", "4096", "-ub", "2048"], "只给 ubatch → batch 兜底 2 倍")
ok(KaiwuManager._prefill_args({"prefill_batch": 1024, "prefill_ubatch": 2048})
   == ["-b", "4096", "-ub", "2048"], "batch < ubatch（非法）→ 钳到 2 倍")

print("\n[10] MoE 专家分配：fit 余量（-fitt）相关判据（2026-10-20 方案）")
# [5] 段的桩改用真实现：本段要验"键按权重路径的量化档算"
KaiwuManager._ctx_cache_key = _REAL_CTX_KEY
# 10a n_part 解析 —— 日志原文格式（scan_fitt_256.log 摘录）
GEMMA_FIT = [
    "0.01.000.000 I srv    load_model: loading model 'D:\\models\\gemma\\gemma-4-26B-A4B-it-UD-IQ3_S.gguf'",
    "0.03.719.084 I common_params_fit_impl: id=0, n_layer=31, n_part=30, overflow_type=4, mem=  5006 MiB",
    "0.03.719.097 I common_params_fit_impl: set ngl_per_device[0].n_layer=31",
    "0.05.444.677 I common_params_fit_impl: id=0, n_layer=31, n_part=24, overflow_type=4, mem=  6604 MiB",
    "0.05.444.691 I common_params_fit_impl: set ngl_per_device[0].(n_layer, n_part)=(31, 24), id_dense_start=0",
    "0.06.318.405 I common_params_fit_impl: set ngl_per_device_high[0].(n_layer, n_part)=(31, 23), id_dense_start_high=0",
    "0.07.223.355 I common_params_fit_impl: set ngl_per_device[0].(n_layer, n_part, overflow_type)=(31, 24, UP), id_dense_start=0",
    "0.08.115.740 I common_params_fit_impl: set ngl_per_device[0].(n_layer, n_part, overflow_type)=(31, 24, GATE), id_dense_start=0",
    "0.09.000.000 I print_info: n_expert              = 128",
    "0.09.000.001 I print_info: n_expert_used         = 8",
    "0.10.092.598 I load_tensors: offloaded 31/31 layers to GPU",
    "0.10.092.602 I load_tensors:   CPU_Mapped model buffer size = 10745.48 MiB",
    "0.10.092.603 I load_tensors:        CUDA0 model buffer size =  3990.03 MiB",
    "0.10.500.000 I llama_context: n_ctx                 = 32768",
    "0.10.600.000 I llama_kv_cache: size =  478.12 MiB ( 4608 cells,  25 layers,  4/1 seqs), K (q8_0):  239.06 MiB, V (q8_0):  239.06 MiB",
    "0.10.700.000 I sched_reserve:      CUDA0 compute buffer size =   186.09 MiB",
]
acct_fit = mgr_with(GEMMA_FIT)._parse_vram_log()
ok(acct_fit is not None and acct_fit.get("n_part") == 24,
   f"-fitt 256 真机日志 → n_part=24（实际 {acct_fit and acct_fit.get('n_part')}）")
ok(acct_fit.get("total_layers") == 31,
   "同段带出 total_layers=31（日志里 n_part=总层数-在显存的专家层数）")
# _high 变体是另一侧的备份方案，**不许采信**：真日志里它报 23，生效的是 24
FIT_HIGH = ["0.01.000.000 I srv    load_model: loading model 'D:\\m\\g-IQ3_S.gguf'",
            "0.05.000.000 I common_params_fit_impl: set ngl_per_device[0].(n_layer, n_part)=(31, 27), id_dense_start=0",
            "0.06.000.000 I common_params_fit_impl: set ngl_per_device_high[0].(n_layer, n_part)=(31, 26), id_dense_start_high=0",
            "0.07.000.000 I common_params_fit_impl: set ngl_per_device[0].(n_layer, n_part, overflow_type)=(31, 27, GATE), id_dense_start=0",
            "0.09.000.000 I print_info: n_expert              = 128",
            "0.10.000.000 I load_tensors: offloaded 31/31 layers to GPU",
            "0.10.100.000 I load_tensors:   CPU_Mapped model buffer size = 10745.48 MiB",
            "0.10.200.000 I load_tensors:        CUDA0 model buffer size =  3990.03 MiB",
            "0.11.000.000 I llama_context: n_ctx                 = 32768",
            "0.11.100.000 I llama_kv_cache: size =  340.00 MiB ( 32768 cells,   5 layers,  4/1 seqs), K (q8_0):  170.00 MiB, V (q8_0):  170.00 MiB",
            "0.11.200.000 I sched_reserve:      CUDA0 compute buffer size =   186.09 MiB"]
ok(mgr_with(FIT_HIGH)._parse_vram_log().get("n_part") == 27,
   "取最后一条生效值、忽略 _high 备份行（实日志：24 生效、_high 报 23）")

# 10b _fit_target_of：与 ctx/moe_offload 同键（模型@量化档）
_td3 = Path(tempfile.mkdtemp(prefix="moefit_"))
(_td3 / "M-IQ3_S.gguf").write_bytes(b"x")
cfg_ft = {"imported_models": {"M": str(_td3 / "M-IQ3_S.gguf")},
          "fit_target": {"M@IQ3_S": 256}, "moe_offload": {"M@IQ3_S": "cpu"}}
ok(KaiwuManager._ctx_cache_key(cfg_ft, "M") == "M@IQ3_S",
   "键 = 模型@量化档（与 ctx_measured / moe_offload 完全同源）")
ok(KaiwuManager._fit_target_of(cfg_ft, "M") == 256, "读到 fit 档位 256")
ok(KaiwuManager._fit_target_of(cfg_ft, "别的模型") is None,
   "没有记录的模型 → None ⇒ 稠密模型/未测过的模型行为完全不变")
ok(KaiwuManager._fit_target_of({"fit_target": {"M@IQ3_S": 256}}, "M") is None,
   "解析不到权重路径时键退化成模型名 → 取不到记录（不张冠李戴）")
ok(KaiwuManager._fit_target_of(dict(cfg_ft, fit_target={"M@IQ3_S": "坏值"}), "M")
   is None, "非法值 → None（绝不拿脏值去改命令行）")

# 10c _vram_pct：扫描的水位保护（>88% 停）
acct_pct = {"model_mib": 6865.0, "kv_mib": 430.0, "compute_mib": 200.0}
pct = m2._vram_pct(acct_pct)
ok(abs(pct - (6865.0 + 430.0 + 200.0) / 8151 * 100) < 0.01,
   f"显存占用 = 权重+KV+计算缓冲 / 总显存（实际 {pct:.1f}%）")
ok(abs(m2._vram_pct(acct_pct) - 0.0) > 80, "gemma -fitt 256 那档水位 92% 级 → 会被水位判据拦")
ok(m2._vram_pct({}) == 0.0, "空账目 → 0（不炸，水位保护自然不触发）")

# 10d 备注话术：fit 切分 ≠ 专家全赶内存（写错就是把结论说错）
acct_n = {"n_expert": 128, "model_mib": 3990.03, "on_gpu": 31,
          "total_layers": 31, "n_part": 24}
n_fit = KaiwuManager._note_moe_hybrid(acct_n, 22.0, 94208, fit=256)
ok(n_fit.startswith("ℹ") and "fit 按份切分" in n_fit and "256" in n_fit
   and "7 层在显存" in n_fit and "22.0 tok/s" in n_fit,
   f"fit 形态备注写清切分/余量/在显存层数/实测速度（实际：{n_fit}）")
n_cpu = KaiwuManager._note_moe_hybrid(acct_n, 20.02, 94208)
ok("mmap 按需流入" in n_cpu and "fit 按份切分" not in n_cpu,
   f"退回 --cpu-moe 时用旧话术，不许说成 fit 切分（实际：{n_cpu}）")

# 10e _measure_tps 的两个统计量：判能不能跑取 max、比档位取 mean
_orig_post3 = _km.requests.post
try:
    _measure_calls.clear()
    _km.requests.post = _fake_seq([14.86, 21.96, 21.65, 21.82])
    ok(m2._measure_tps(CFG, rounds=4) == 21.96,
       "max（默认）：判'这档能不能跑'，偶发高值不被均值摊薄")
    _measure_calls.clear()
    _km.requests.post = _fake_seq([14.86, 21.96, 21.65, 21.82])
    vmean = m2._measure_tps(CFG, rounds=4, stat="mean")
    ok(abs(vmean - (21.96 + 21.65 + 21.82) / 3) < 1e-9,
       f"mean：比档位优劣（预热外 3 枪取均值，实际 {vmean:.3f}）")
    _measure_calls.clear()
    _km.requests.post = _fake_seq([14.86, 21.96, 21.65, 21.82])
    m2._measure_tps(CFG, rounds=4)
    ok(len(_measure_calls) == 4, "打 4 枪（第 1 枪是预热，剩 3 枪才是有效样本）")
    # 独立验收报告：MoE 权重走 mmap 时前 4~5 枪都在爬坡，7 枪必须丢前 2
    _measure_calls.clear()
    _km.requests.post = _fake_seq([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    v7 = m2._measure_tps(CFG, rounds=7, stat="mean")
    ok(abs(v7 - (3 + 4 + 5 + 6 + 7) / 5) < 1e-9,
       f"7 枪丢前 2、取后 5 均值（爬坡枪不参与比较；实际 {v7:.2f}）")
    _measure_calls.clear()
    _km.requests.post = _fake_seq([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    v7d = m2._measure_tps(dict(CFG, ctx_tps_drop=0), rounds=7, stat="mean")
    ok(abs(v7d - 4.0) < 1e-9, f"ctx_tps_drop 显式配置优先（0 → 全 7 枪均值 4.0）")
    _measure_calls.clear()
    _km.requests.post = _fake_seq([1.0, 2.0, 3.0])
    v3 = m2._measure_tps(CFG, rounds=3, stat="mean")
    ok(abs(v3 - 2.5) < 1e-9, "不足 5 枪时只丢第 1 枪（样本太少，丢多了没法比）")
finally:
    _km.requests.post = _orig_post3

# 10f DEFAULTS 与当前二进制
ok(DEFAULTS.get("ctx_tps_tokens") == 400 and DEFAULTS.get("ctx_tps_rounds") == 7
   and DEFAULTS.get("ctx_tps_drop") == 2,
   f"DEFAULTS：400 token / 7 枪 / 丢前 2（跨加载噪声实测 1.19，"
   f"4 枪 1 丢的读数没有分辨力；实际 {DEFAULTS.get('ctx_tps_tokens')}/"
   f"{DEFAULTS.get('ctx_tps_rounds')}/{DEFAULTS.get('ctx_tps_drop')}）")
ok(DEFAULTS.get("fit_target") == {},
   "DEFAULTS 带 fit_target 空表（稠密模型永远不写它 ⇒ 行为零变化）")
from app.config import resolve_llama_server
_ls_exe = resolve_llama_server()
ok((not _ls_exe) or m2._llama_supports({"llama_server_exe": _ls_exe}, "-fitt") is True,
   "当前 llama-server 认 -fitt（不认时 _start_inner 与扫描会自动退回 --cpu-moe 旧逻辑）")

print("\n[11] _start_inner 参数选择：fit 优先、老记录退回、extra_args 不叠加（坑 1）")
# 参数链 start → _start_inner → 命令行是"静默失效"的高发区（方案 §6.1 坑 1）：
# 漏一层就是"扫描跑了但旗标没生效"，日志里只会觉得"档位没差别"。这里拦住。
# 不起进程：命令行日志在 Popen 之前就打好了，让 Popen 故意炸掉即可读到它。
_td4 = Path(tempfile.mkdtemp(prefix="moeargs_"))
(_td4 / "G-IQ3_S.gguf").write_bytes(b"x")
_orig_popen = _km.subprocess.Popen
_orig_probe = KaiwuManager.probe
try:
    def _boom(*a, **k):
        raise RuntimeError("stop-here")      # 参数已进日志，这里故意炸
    _km.subprocess.Popen = _boom
    KaiwuManager.probe = lambda self, cfg=None, timeout=1.0: (False, [])
    m3 = KaiwuManager()
    m3._llama_supports = lambda cfg, flag: True        # 假装二进制认 -fitt/--cpu-moe
    base = {"port": 11435, "host": "127.0.0.1", "llama_server_exe": _ls_exe,
            "imported_models": {"G": str(_td4 / "G-IQ3_S.gguf")}}

    def _cmd(cfg, **kw):
        m3._logbuf.clear()
        m3._start_inner("G", cfg, timeout=1, ctx_override=32768, **kw)
        return next((l[2:] for l in m3._logbuf if l.startswith("$ ")), "")

    c1 = _cmd(dict(base, fit_target={"G@IQ3_S": 256},
                   moe_offload={"G@IQ3_S": "cpu"}))
    ok("-fitt 256" in c1 and "--cpu-moe" not in c1,
       f"有 fit 记录 → 传 -fitt 256 且**绝不**再带 --cpu-moe（否则新档位不生效）")
    c2 = _cmd(dict(base, moe_offload={"G@IQ3_S": "cpu"}))
    ok("-fitt" not in c2 and "--cpu-moe" in c2,
       "只有老 moe_offload 记录（没 fit）→ 退回 --cpu-moe（回滚路径原样保留）")
    c3 = _cmd(dict(base, fit_target={"G@IQ3_S": 256}), extra_args=["-fitt", "512"])
    ok(c3.endswith("-fitt 512") and c3.count("-fitt") == 1,
       f"extra_args（调用方显式旗标）不叠加配置记录，避免两个 -fitt 打架")
    c4 = _cmd(base)
    ok("-fitt" not in c4 and "--cpu-moe" not in c4,
       "稠密模型 / 没任何记录 → 两个旗标都不加 ⇒ 行为零变化")
    c5 = _cmd(dict(base, fit_target={"G@IQ3_S": 256}),
              extra_args=["--cpu-moe"])
    ok("--cpu-moe" in c5 and "-fitt" not in c5,
       "fit 扫描全部失败时 attempt(moe_cpu=True) 仍能显式要 --cpu-moe")
finally:
    _km.subprocess.Popen = _orig_popen
    KaiwuManager.probe = _orig_probe

print("\n[12] measure_ctx 全流程（桩驱动）：扫描之后每次加载都必须带同一个 -fitt")
# 2026-10-20 真机验收抓到的坑（方案 §6.1 坑 1 的实例）：7 处 attempt 调用点漏了
# "估值验证"那一处 → 122880 是在**默认卸载**下验的，配上 -fitt 2048 真跑掉 8 层、
# 稳态只有 17.3 tok/s（比改前的 --cpu-moe 还差 25.6）。这种"静默失效"在日志里
# 只表现为"档位没差别"，只能靠逐次记录 start 收到的旗标来锁死。
_td5 = Path(tempfile.mkdtemp(prefix="moeflow_"))
(_td5 / "G-IQ3_S.gguf").write_bytes(b"x")
_orig_sleep = _km.time.sleep
_km.time.sleep = lambda *a, **k: None          # attempt() 每次 sleep 1s，桩里跳过


def _fitt_of(call):
    e = call.get("extra") or []
    return int(e[1]) if len(e) == 2 and e[0] == "-fitt" else None


def _sim_measure(tps_fn, hot_gt=94208, npart_fn=None):
    """桩驱动跑一遍 measure_ctx → (结果, 每次 start 的旗标, 落盘记录, 日志行)。"""
    mgr = KaiwuManager()
    mgr._logbuf.clear()
    state = {"ctx": 0}
    starts, records = [], []
    cfg = dict(CFG, ctx_probe_ctx=32768, ctx_auto_min=32768,
               ctx_auto_max=131072, ctx_bisect_max_loads=3,
               imported_models={"G": str(_td5 / "G-IQ3_S.gguf")},
               ctx_tps_tokens=400, ctx_tps_rounds=7)
    mgr.stop = lambda cfg=None: True
    mgr._llama_supports = lambda cfg2, flag: flag in ("-fitt", "--cpu-moe")

    def _start(model, cfg2=None, timeout=None, ctx_override=None,
               force_measure=False, moe_cpu=False, extra_args=None):
        state["ctx"] = int(ctx_override or 0)
        starts.append({"ctx": state["ctx"], "moe_cpu": moe_cpu,
                       "extra": list(extra_args) if extra_args else None})
        return True

    def _acct():
        # 真机数值：权重 3.7G + 计算缓冲 896 / 每 token KV 10.6 KiB
        # → _estimate_ctx 给 122880（与 2026-10-20 真机日志一致）；
        # on_gpu：窗口 > 94208 时掉到 23/31（模拟真机"大窗口配 fit 切分"掉层）
        ctx = state["ctx"]
        return {"n_expert": 128, "n_expert_used": 8, "print_info_seen": True,
                "model_mib": 3693.0, "compute_mib": 896.0,
                "cpu_mapped_mib": 10745.48, "kv_mib": ctx * 10.6 / 1024.0,
                "kv_fixed_mib": 0.0, "per_token_kv_mib": 10.6 / 1024.0,
                "n_ctx": ctx, "on_gpu": 31 if ctx <= hot_gt else 23,
                "total_layers": 31,
                "n_part": (npart_fn(_fitt_of(starts[-1]))
                           if npart_fn else 24)}

    mgr.start = _start
    mgr._gpu_offload_info = lambda: (31 if state["ctx"] <= hot_gt else 23, 31)
    mgr._parse_vram_log = _acct
    mgr._measure_tps = (lambda cfg2, tokens=None, timeout=None, rounds=None,
                        stat="max", drop=None:
                        tps_fn(_fitt_of(starts[-1]), state["ctx"]))
    mgr._save_measured = (lambda cfg2, model, ctx, note="", mode="", fit=None:
                          records.append({"ctx": ctx, "fit": fit, "mode": mode}))
    mgr.granted_ctx = lambda cfg2=None: state["ctx"]
    res = mgr.measure_ctx("G", cfg)
    return res, starts, records, list(mgr._logbuf)


try:
    # ---- 12a 主流程：扫描 4096→2048→1024 后停，取最快的 2048 ----
    _T1 = {4096: 23.0, 2048: 24.41, 1024: 22.63, None: 19.5}
    res1, st1, rec1, lg1 = _sim_measure(lambda f, ctx: _T1.get(f, 20.0))
    scans = [c for c in st1 if c["ctx"] == 32768 and c["extra"]]
    ok([c["extra"] for c in scans] == [["-fitt", "4096"], ["-fitt", "2048"],
                                       ["-fitt", "1024"]],
       f"扫描逐档实测并按边际收益停（实际 {[c['extra'] for c in scans]}）")
    ok(st1[0]["extra"] is None and st1[0]["moe_cpu"] is False,
       "第 0 步探测不带 MoE 旗标（那时还不知道该选哪档）")
    ok(all(c["extra"] for c in st1[1:]),
       f"⚠**扫描之后每一次加载都带 -fitt**（漏一次就是「档位没差别」+ 记错窗口，"
       f"实际 {[c['extra'] for c in st1[1:]]}）")
    ok(rec1 and rec1[0]["fit"] == 2048 and rec1[0]["mode"] == "",
       f"选中的 2048 写进 fit_target，且不再写 mode=cpu（实际 {rec1}）")
    ok(rec1[0]["ctx"] == res1["best"] and rec1[0]["ctx"] > 32768,
       f"记录窗口 = 测量结论（{rec1[0]['ctx']}），且大于探测窗口")
    ok(st1[-1]["extra"] == ["-fitt", "2048"] and st1[-1]["ctx"] == rec1[0]["ctx"],
       f"最后一次加载就是**记录形态**（同窗口同 -fitt），服务停在正确形态"
       f"（实际 {st1[-1]}）")
    ok(any("最终窗口" in l and "保持 2048" in l for l in lg1),
       "§6.2 在最终窗口下复核了相邻档，并以「没更快」为由保持选中档")
    ok(any("n_part=24" in l for l in lg1), "扫档日志带出 n_part（辅助判据可见）")

    # ---- 12b §6.2 的替换分支：更激进档只在最终窗口下更快 → 改用它 ----
    def _T2(f, ctx):
        if ctx <= 32768:
            return {4096: 23.0, 2048: 24.41, 1024: 22.63, None: 19.5}.get(f, 20.0)
        return {2048: 22.0, 1024: 26.5}.get(f, 20.0)

    res2, st2, rec2, lg2 = _sim_measure(_T2)
    ok(rec2[0]["fit"] == 1024,
       f"最终窗口下 1024（26.5）真的比 2048（22.0）快 → 改用 1024（实际 {rec2}）")
    ok(st2[-1]["extra"] == ["-fitt", "1024"],
       "替换后不需要「再回原档」（省一次加载），服务停在改选后的形态")
    ok(any("改用更激进档 1024" in l for l in lg2), "日志明说改档原因（实测数对比）")

    # ---- 12c 独立验收报告的稳态曲线：n_part 停滞才停，同分取更保守 ----
    # 报告 §三（32768 窗口、顺序打乱、9 枪丢前 4）：
    #   4096→18.82(0层) 2048→30.50(4层) 1024→33.80(8层) 512→34.35(10层)
    #   256→35.46(11层) 128→35.95(11层)
    # 旧逻辑（阈值 0.3、4 枪丢 1）在这里停在 512；新逻辑必须走到底再回选 256：
    # 1024→512 只 +0.55、256→128 只 +0.49，都小于跨加载噪声 1.2，
    # **只有补上「n_part 不再下降」判据才可能继续往后扫**。
    _NP = {4096: 31, 2048: 27, 1024: 24, 512: 22, 256: 20, 128: 20, None: 31}
    _T3 = {4096: 18.82, 2048: 30.50, 1024: 33.80, 512: 34.35,
           256: 35.46, 128: 35.95}

    def _tps3(f, ctx):
        if ctx > 32768:                      # 大窗口下选中的 256 稳定 31.4 级
            return 31.4 if f == 256 else 30.0
        return _T3.get(f, 20.0)

    res3, st3, rec3, lg3 = _sim_measure(_tps3, npart_fn=lambda f: _NP.get(f, 31))
    scanned = [c["extra"][1] for c in st3
               if c["ctx"] == 32768 and c["extra"] and c["extra"][0] == "-fitt"]
    ok(scanned == ["4096", "2048", "1024", "512", "256", "128"],
       f"n_part 还在下降时不因噪声级增益提前收工（实际扫了 {scanned}）")
    ok(any("扫档前预热" in l for l in lg3),
       "扫档前先打预热枪（否则后跑的档天然更热 → 系统性抬高读数）")
    ok(any("n_part 不再下降" in l for l in lg3),
       "停止判据写明是 n_part 停滞（不是只看速度）")
    ok(rec3[0]["fit"] == 256,
       f"128(35.95) 与 256(35.46) 差 0.49 < 噪声 1.2 → 取更保守的 256"
       f"（实际 {rec3[0]['fit']}）")
    ok(res3["best"] == rec3[0]["ctx"] > 32768,
       f"窗口记录跟着走（实际 {res3['best']}）")
    _km.time.sleep = _orig_sleep
except Exception as e:
    _km.time.sleep = _orig_sleep
    raise

print(f"\n===== {PASS} 通过 / {FAIL} 失败 =====")
if FAIL:
    print("注：任一 FAIL 都意味着 MoE 又被冤判/备注又开始编数 —— 对照文件头部的真机数据。")
sys.exit(1 if FAIL else 0)
