"""CPU_Mapped 判据的 Ornith 形态回归（2026-10-16，WB→wb2 P0 证据测试）。

背景：wb2 的 gemma 修复给 _weights_fully_resident 加了
CPU_Mapped ≤ 10% 判据（gemma-4-26B 实测 76% → 1.8 tok/s，判 False 正确）。
但 Ornith-1.5-9B@Q4_K_M 在 32768 全层档的**真实加载日志**（人类
17:5x console 全量 scrollback，本测试原样引用）是：

    offloaded 33/34 layers to GPU
    CPU_Mapped model buffer size =   683.55 MiB
    CUDA0 model buffer size =  4674.33 MiB

占比 683.55/(683.55+4674.33) = 12.8% > 10% ⇒ 现判据误判 False。
但那 683 MiB 是**冷数据**：blk.32 nextn 幽灵块（15 个张量被 --ignoring，
约 145 MiB，_gpu_offload_info 已正确豁免）+ token 嵌入表（约 538 MiB，
每 token 只查一行）。实测该档 43.28 t/s 全速，真实计算层 33/33 全在 GPU。

规格（本测试即规格）：
  1. _gpu_offload_info 必须豁免幽灵块 → (33, 33)          （现已正确）
  2. _parse_vram_log 必须取到主模型段账目                  （现已正确）
  3. _weights_fully_resident 必须认这个已知好档 → True     （⭐ 现误判 False，待修）
  4. gemma 形态（76%）必须判 False                        （wb2 已覆盖，防回归）

修法方向（wb2 定）：把冷数据从 CPU_Mapped 里剔除再算占比——
W 行 ignored 张量size 可精确求和；嵌入表 ≈ n_vocab×n_embd×bpw，
或对"层缺口==幽灵块数"的档直接豁免。
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


# ---- 人类 console 日志原样摘录（Ornith-1.5-9B-Q4_K_M @ 32768 全层档） ----
ORNITH = [
    "0.00.172.959 I srv    load_model: loading model 'D:\\models\\Ornith-1.5-9B\\Ornith-1.5-9B-Q4_K_M.gguf'",
    "0.01.640.146 I print_info: n_ctx_train           = 262144",
    "0.01.640.147 I print_info: n_layer               = 32",
    "0.01.640.148 I print_info: n_layer_all           = 33",
    "0.01.640.217 I load_tensors: loading model tensors, this can take a while... (load_mode = mmap)",
    "0.01.642.563 W model has unused tensor blk.32.attn_norm.weight (size = 16384 bytes) -- ignoring",
    "0.01.642.573 W model has unused tensor blk.32.post_attention_norm.weight (size = 16384 bytes) -- ignoring",
    "0.01.642.590 W model has unused tensor blk.32.attn_q.weight (size = 18874368 bytes) -- ignoring",
    "0.01.642.624 W model has unused tensor blk.32.attn_k.weight (size = 2359296 bytes) -- ignoring",
    "0.01.642.631 W model has unused tensor blk.32.attn_v.weight (size = 3440640 bytes) -- ignoring",
    "0.01.642.654 W model has unused tensor blk.32.attn_output.weight (size = 9437184 bytes) -- ignoring",
    "0.01.642.661 W model has unused tensor blk.32.attn_q_norm.weight (size = 1024 bytes) -- ignoring",
    "0.01.642.667 W model has unused tensor blk.32.attn_k_norm.weight (size = 1024 bytes) -- ignoring",
    "0.01.642.674 W model has unused tensor blk.32.ffn_gate.weight (size = 28311552 bytes) -- ignoring",
    "0.01.642.681 W model has unused tensor blk.32.ffn_down.weight (size = 41287680 bytes) -- ignoring",
    "0.01.642.687 W model has unused tensor blk.32.ffn_up.weight (size = 28311552 bytes) -- ignoring",
    "0.01.642.694 W model has unused tensor blk.32.nextn.eh_proj.weight (size = 18874368 bytes) -- ignoring",
    "0.01.642.701 W model has unused tensor blk.32.nextn.enorm.weight (size = 16384 bytes) -- ignoring",
    "0.01.642.710 W model has unused tensor blk.32.nextn.hnorm.weight (size = 16384 bytes) -- ignoring",
    "0.01.642.730 W model has unused tensor blk.32.nextn.shared_head_norm.weight (size = 16384 bytes) -- ignoring",
    "0.02.018.877 I load_tensors: offloading output layer to GPU",
    "0.02.018.882 I load_tensors: offloading 32 repeating layers to GPU",
    "0.02.018.883 I load_tensors: offloaded 33/34 layers to GPU",
    "0.02.018.886 I load_tensors:   CPU_Mapped model buffer size =   683.55 MiB",
    "0.02.018.888 I load_tensors:        CUDA0 model buffer size =  4674.33 MiB",
    "0.03.502.299 I llama_context: constructing llama_context",
    "0.03.502.300 I llama_context: n_ctx                 = 32768",
    "0.03.504.297 I llama_kv_cache:      CUDA0 KV buffer size =   544.00 MiB",
    "0.03.516.979 I llama_kv_cache: size =  544.00 MiB ( 32768 cells,   8 layers,  4/1 seqs), K (q8_0):  272.00 MiB, V (q8_0):  272.00 MiB",
    "0.03.518.949 I llama_memory_recurrent:        CPU RS buffer size =     8.38 MiB",
    "0.03.524.387 I llama_memory_recurrent:      CUDA0 RS buffer size =   192.62 MiB",
    "0.03.543.431 I sched_reserve:      CUDA0 compute buffer size =   239.66 MiB",
    "0.03.760.223 I srv  llama_server: model loaded",
]

# ---- gemma-4-26B@IQ3_S 病态形态（wb2 实证：76% 在内存流，1.8 tok/s） ----
GEMMA = [
    "0.00.172.959 I srv    load_model: loading model 'D:\\models\\gemma-4-26B-A4B-it-UD\\gemma-4-26B-A4B-it-UD-IQ3_S.gguf'",
    "0.01.640.146 I print_info: n_ctx_train           = 262144",
    "0.02.018.883 I load_tensors: offloaded 31/31 layers to GPU",
    "0.02.018.886 I load_tensors:   CPU_Mapped model buffer size = 10745.00 MiB",
    "0.02.018.888 I load_tensors:        CUDA0 model buffer size =  3407.00 MiB",
    "0.03.502.300 I llama_context: n_ctx                 = 102400",
    "0.03.504.297 I llama_kv_cache:      CUDA0 KV buffer size =   340.00 MiB",
    "0.03.516.979 I llama_kv_cache: size =  340.00 MiB ( 32768 cells,   5 layers,  4/1 seqs), K (q8_0):  272.00 MiB, V (q8_0):  272.00 MiB",
    "0.03.543.431 I sched_reserve:      CUDA0 compute buffer size =   239.66 MiB",
]


def load_with(lines):
    mgr = KaiwuManager()
    mgr._logbuf.clear()
    mgr._logbuf.extend(lines)
    return mgr


print("\n[1] Ornith 32768 全层档（真实日志）")
mgr = load_with(ORNITH)
info = mgr._gpu_offload_info()
ok(info == (33, 33), f"幽灵块豁免：33/34 缺口 1 == ignored 1 ⇒ (33,33)（实际 {info}）")
acct = mgr._parse_vram_log()
ok(acct is not None, "主模型段账目解析成功")
if acct:
    ok(abs(acct.get("cpu_mapped_mib", 0) - 683.55) < 0.01,
       f"CPU_Mapped=683.55（实际 {acct.get('cpu_mapped_mib')}）")
    ok(abs(acct.get("model_mib", 0) - 4674.33) < 0.01,
       f"CUDA0 model buffer=4674.33（实际 {acct.get('model_mib')}）")
    ratio = acct["cpu_mapped_mib"] / (acct["cpu_mapped_mib"] + acct["model_mib"])
    print(f"  ..  占比 = {ratio:.1%}（判据线 10%）")
    ok(mgr._weights_fully_resident(acct) is True,
       "已知好档必须判 True（43.28 t/s 实证；683 MiB=幽灵块145+嵌入表约538，均冷数据）")

print("\n[2] gemma 102400 病态档（wb2 实证日志形态）")
mgr = load_with(GEMMA)
acct_g = mgr._parse_vram_log()
ok(acct_g is not None and abs(acct_g.get("cpu_mapped_mib", 0) - 10745.0) < 0.01,
   f"CPU_Mapped=10745（实际 {acct_g and acct_g.get('cpu_mapped_mib')}）")
ok(mgr._weights_fully_resident(acct_g) is False, "病态档必须判 False（76% 内存流）")

print(f"\n===== {PASS} 通过 / {FAIL} 失败 =====")
if FAIL:
    print("注：[1] 的 resident 断言失败即 wb2 P0 误判的实证——")
    print("    该断言转绿前，带 nextn 幽灵块的 qwen35 模型每档都会被冤杀。")
sys.exit(1 if FAIL else 0)
