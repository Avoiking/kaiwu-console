"""Kaiwu 控制台 - 配置与路径管理。

数据全部落在应用目录下的 data/，保证绿色便携：
  data/config.json    全局配置（含 imported_models 模型映射、last_model）

⚠️ 本文件里的 DEFAULTS **不含任何本机私人路径** —— 默认值必须让任何人的
   干净克隆跑得起来。所有跟"我这台机器"绑定的值都走下面 `resolve_llama_server()`
   的自动发现，或由用户填进 data/config.json（示例见 config.example.json）。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

APP_NAME = "Kaiwu Console"


def app_dir() -> Path:
    """应用根目录：打包后为 exe 所在目录，开发时为本文件上一级。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    d = app_dir() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_path() -> Path:
    return data_dir() / "config.json"


DEFAULTS: dict = {
    # ---- 推理服务（直连 llama-server，OpenAI 兼容端点）----
    "host": "127.0.0.1",
    "port": 11435,
    # llama-server 可执行文件的路径。**留空 = 自动发现**（见 resolve_llama_server）：
    # 会依次找 环境变量 KAIWU_LLAMA_SERVER → 仓库内 bin/ → 仓库内 llama-deps 的同级/
    # 上级目录 → ~/.kaiwu/bin*/ → PATH。
    # 实在找不到就把这里的值填成完整路径，例如
    #   "D:\\AI-Tools\\llama.cpp\\build\\bin\\Release\\llama-server.exe"
    # 提示：llama-server **不随本仓库分发**（二进制体积大且有平台差异），
    # 请从 https://github.com/ggml-org/llama.cpp/releases 下载官方 Windows 包。
    "llama_server_exe": "",
    "startup_timeout_sec": 900,     # 首次加载（50 系 CUDA 编译）可能很慢，给足时间
    "context_size": "",             # 留空 = 按模型覆盖/测量缓存/默认 32768，fit 自动降
    # ---- 自动测量上下文（把旧的"基准测试定 ctx"以更稳的方式做回来）----
    # 旧的 kaiwu CLI 是每次启动都跑 warmup 基准，结果是加载慢 + 8G 显存上误判死循环。
    # 现在改成"探测一次 → 结果缓存 → 以后自动采用"：
    # ctx_measured: 测量结果 {模型@量化档: 每槽 n_ctx}，由 KaiwuManager.measure_ctx()
    #   写入；_target_ctx 会优先采用它。键带量化档（2026-10-15 起）：不同量化
    #   体积不同、能开的窗口不同，一个模型每个档各记一条。文件无量化后缀时
    #   键就是模型名本身（兼容旧配置的平名记录——仅该模型只剩一个量化档时
    #   采信，多档共存时旧记录可能测自任何一档，不予采信）。
    #   显式手填的 context_size / model_ctx_overrides 数字仍然优先于缓存。
    "ctx_measured": {},
    # ctx_measured_env: 测量时的环境指纹 {模型@量化档: {gpu, vram_mib, kv, llama}}。
    #   **不要手填**。每次加载前会比对：显卡换了 / KV 量化档改了 / llama-server
    #   换版本了 → 旧缓存作废、自动重测。没这份指纹的旧缓存一律视为不可信
    #   （开源场景下尤其重要：你的 131072 未必是别人的 131072）。
    "ctx_measured_env": {},
    # ctx_measured_note: 测量结论的说明 {模型@量化档: 文本}。degraded
    #   （权重装不下显存、只能按下限记录）时由 measure_ctx 写入，UI 卡片
    #   直接展示给用户；正常测量会清掉对应键。**不要手填**。
    "ctx_measured_note": {},
    # ctx_auto: 无显式值、且该模型当前量化档还没有**有效**测量缓存时，加载前
    #   自动探测一次。2026-10-07 起默认 **true** —— 因为 measure 从"最多 10+ 次
    #   真实加载"改成了"1 次读账目 + 至多 1 次验证"（见 kaiwu_manager.measure_ctx），
    #   首次只多等两次加载，测完写进 ctx_measured 就永不再测（同量化档）。
    #   想完全关掉自动探测就设 false，或给某个模型写死数字/写 "auto" 单独控制。
    "ctx_auto": True,
    "ctx_auto_max": 131072,         # 二分兜底时的上界（每槽）
    "ctx_auto_min": 32768,          # 下限，比这还低就落回默认 32768
    # ctx_probe_ctx: 自动测量时第一次加载用的窗口。只需要够读出日志里的显存
    #   账目（权重/KV/计算缓冲各多少 MiB），不需要大。调大只会让首次探测更慢，
    #   而且过大的 KV 分配本身可能挤掉层，把"本来能开"测成"开不了"。
    "ctx_probe_ctx": 32768,
    # ctx_vram_reserve_mib: 算式估算时预留给驱动上下文/显存碎片/多 slot 的
    #   安全余量（MiB）。
    "ctx_vram_reserve_mib": 512,
    # ctx_estimate_ratio: 算式估算结果的校准系数。**纯算式一定会高估**（本机
    #   实测高估 1.7~2.3 倍，因为 llama.cpp 在显存占用 ~94% 时不报错、而是悄悄
    #   把层赶到 CPU，且日志里的三个 buffer 数不含 CUDA context 本身）。
    #   0.42 = 本机两次实测里较保守的值，留足安全垫。实测效果：4B-ud 估出
    #   106496 正好全层在 GPU，距真实临界 110592 只差 4%。
    #   ⚠️ 猜错的表现是「掉层到 CPU、速度掉十倍」，不是报错，所以别贪。
    "ctx_estimate_ratio": 0.42,
    # ctx_bisect_max_loads: 估值命中后还要不要继续往上顶，把窗口逼到**真正**
    #   的临界点。0 = 只验证估值那一次（多数情况够用）；3 = 命中后再顶一两档
    #   试试（本机实测 3 次拿到 106496）；10 = 旧行为，完整二分追到极限。
    #   只数"基线定下来之后"的加载（估值验证 + 二分）—— 探测（MoE 还有一次
    #   --cpu-moe 探测）是读账目的固定成本，不吃这个预算。
    #   每次加载都要真实启停服务（本机 4B 约 7s，大模型更久），按需选。
    "ctx_bisect_max_loads": 3,
    # ---- MoE 模型的混合卸载测量（小显存玩大 MoE，2026-10-17）----
    # MoE（混合专家）模型权重常常远超显存（如 gemma-4-26B-A4B 15.1G vs 8G 卡）。
    # 这不是"装不起"：每 token 只读激活的几个专家，冷专家躺在内存 mmap 里
    # 不拖速度。控制台对这类模型的处理：
    #   ctx_moe_prefer_cpu: 未全驻时优先试 --cpu-moe（专家权重显式赶去内存，
    #     显存全留给 KV/计算）。实测同窗口比默认卸载更快、窗口能翻几倍；
    #     关掉则只用 llama.cpp 默认的自动 fit（fit 到顶也会 CPU_Mapped）。
    #   ctx_moe_min_tps: MoE 档位的实测速度门槛（tok/s）。低于它的窗口再大
    #     也不算"可持续"—— 2026-10-17 实证：31/31 层都在 GPU 也可能只有
    #     1.18 tok/s（显存贴顶），层数判据看不出来，只有真测能抓住。
    #     默认 15（2026-10-17 定）：要同时拦住两种坏档 ——
    #     ① 病态档：显存贴顶的 1.18 tok/s（5 的旧值就为它设的）；
    #     ② 悬崖档：gemma --cpu-moe @196608 实测掉到过 8.2（贴边档，
    #        同档几小时前还读到 25.0，速度不稳），而 184320 有 27.3。
    #     不设 20/30：整合包要发到别人的机器上，弱一档的卡跑 gemma 级
    #     MoE 稳定也就 15~18，门槛高了会把它们冤到窗口下限。15 是
    #     "明显不可用"和"弱机器甜点位"之间的线。
    #   ctx_tps_tokens / ctx_tps_timeout_sec: 测速用的生成请求规模/超时。
    #     tokens 默认 400（2026-10-20 从 48 改过来）：fit 档位扫描要**比档位
    #     优劣**，而档位间的差距只有 0.3~1.2 —— 48 token 的均值极差能到 0.38，
    #     噪声盖过信号会选错档。400 token 均值极差降到 0.11。**不能用 300**
    #     （实测均值还低 1.3%）。
    #   ctx_tps_rounds / ctx_tps_drop: **每次测速打几枪、丢前几枪**。
    #     默认 7 枪 / 丢前 2（2026-10-20 晚按独立验收报告从 3/1、4/1 改过来）：
    #     独立复测发现 MoE 专家权重走内存 mmap 时**前 4~5 枪都还在爬坡**
    #     （页缓存逐渐喂热，16G 内存装不下 11.3G 权重）——「丢 1 剩 2~3」
    #     拿到的全是冷态读数：
    #         32768 + fitt512 连打：29.90 → 32.88 → 33.65 → 34.34 → 34.46 → 34.60 → 34.56
    #         94208 + fitt256 连打：18.59 → 24.30 → 29.29 → 30.11 → 30.63 → 31.57 → 32.26（+74%）
    #     丢前 2、取后 5 才是稳态。代价：每档多 2~3 枪（约 +40s），而测量
    #     每个模型@量化档一生只跑一次。
    #     丢掉预热后的统计量分工见 _measure_tps：判「能不能跑」取最大值，
    #     比「档位优劣」取均值。
    "ctx_moe_prefer_cpu": True,
    "ctx_moe_min_tps": 15.0,
    # ctx_dense_min_tps: **稠密**模型的实测速度门槛（tok/s），与 MoE 分开定。
    #   为什么分开：两种模型的物理上限差一个量级，混用一个门槛必错一头。
    #   · 稠密小模型（9B Q4 级）全驻显存时 30+ 是常态，掉几层到内存就腰斩
    #     （本机实测 Ornith @114688：19/34 层、45% 权重走 PCIe → 12.5 tok/s）。
    #     用户 2026-10-17 拍板：小的稠密模型至少 30，达不到就换更小窗口，
    #     不拿半速糊弄用户。
    #   · MoE（26B-A4B 级）每 token 只读激活的几个专家，8G 卡上 20~26 就是
    #     甜点位；门槛 5 只负责抓显存贴顶的病态档（实测过 1.18 tok/s），
    #     按 30 判它永远过不了，窗口会被冤到下限。
    #   门槛只卡"掉层/混合"档；**全驻档不看速度直接过**——全驻已是该模型
    #   在当前卡上最快的形态，没有更好的了，再卡速度只会把所有大模型冤到下限。
    "ctx_dense_min_tps": 30.0,
    "ctx_tps_tokens": 400,
    "ctx_tps_timeout_sec": 120,
    "ctx_tps_rounds": 7,
    "ctx_tps_drop": 2,
    # fit_gain_min_tps: **速度差容差**（tok/s）——界面上叫这个名字，不叫"噪声"。
    #   含义：扫档时两档测速相差小于它，就当作"一样快"，停止继续往下试，
    #   并在几个"一样快"的档里选**更省显存**的那个。
    #   为什么需要它：同一台机器同一个配置重复加载，读数本身就会差一点
    #   （本机实测 1.19 tok/s，两个 n_part 与显存完全相同的档测出 32.02 与 33.21）。
    #   比这个差还小的"提升"分不出来，当作没有，免得为抖动多占显存。
    #   默认 1.2 = 开发机（8G 显存笔记本 + gemma-4-26B IQ3_S）实测的跨加载差值。
    #   ⚠️ 换机器/换模型后这个值会变：速度快的小模型绝对值也大，大显存机器能放的
    #      层多、波动也大。分不清时**填大一点更安全**（不容易被抖动骗着改档）。
    "fit_gain_min_tps": 1.2,
    # moe_offload: {模型@量化档: "cpu"} 测量选定的 MoE 卸载模式，**不要手填**。
    #   记了 "cpu" 的档位每次加载都会带 --cpu-moe（与测量时的形态保持一致，
    #   否则测出的窗口换套卸载行为不一定还成立）；全驻测量成功会清除该键。
    #   ⚠️ 2026-10-20 起它是**回滚路径**：有 fit_target 记录时优先级更低
    #   （见 fit_target），只有"扫描全失败才退回 --cpu-moe"的档位才记它。
    "moe_offload": {},
    # fit_target: {模型@量化档: MiB} —— MoE 专家分配的最优 fit 余量，
    #   测量时自动扫描选出并写入，**不要手填**。键与 ctx_measured 同源
    #   （_ctx_cache_key）。加载时给 llama-server 传 `-fitt <MiB>`，让 fit
    #   自己按份切分专家权重（每层内部均分）—— 实测同窗口比 --cpu-moe
    #   全下放快 9.7%（gemma-4-26B IQ3_S）/ 22.1%（LFM2.5-24B），窗口不减，
    #   多用掉的显存本来就在空着。
    #   ⚠️ 有记录 ⇒ 该模型是 MoE 且已实测过 fit 切分更优 ⇒ **绝不再传
    #   --cpu-moe**（否则新档位不生效）。稠密模型永远不会有这条记录，
    #   行为完全不变。换显卡 / 改 KV 量化 / 换二进制 / 换量化档时，
    #   随 ctx_measured_env 指纹一起作废重测（复用同一份指纹，不另设一套）。
    "fit_target": {},
    # ---- 预填批量（长会话第一轮的全量预填提速，2026-10-17 实测）----
    # llama.cpp 默认 -b 2048 -ub 512。长会话每一轮的第一遍（冷缓存）要把
    # 整段历史全量预填，默认批量下 GPU 利用率很低。实测（gemma-4-26B-A4B
    # IQ3_S，--cpu-moe @98304，90K 提示，8G 独显）：
    #   默认 2048/512   → 预填  401 tok/s（90K ≈ 190s）
    #   -b 4096 -ub 2048 → 预填  908 tok/s（≈84s，2.26 倍），解码不变
    #   -b 8192 -ub 4096 → 预填  965 tok/s，但再多占 1.2G 显存，不值
    # 代价：compute buffer 约 +700 MiB —— 显存紧的模型窗口会小一点。但 ctx
    # 自动测量读的就是真实加载的账目（compute_mib 在算式里），这点它自己
    # 会算进去，不用你操心。想恢复 llama.cpp 默认就把 ubatch 设成 512。
    "prefill_batch": 4096,
    "prefill_ubatch": 2048,
    # ---- KV 缓存量化 ----
    # 默认 q8_0（2026-10-17 假新机测试后从空=f16 改过来）。空值让没配过的新
    # 机器拿 f16 KV —— 同机同模型实测（Ornith-1.5-9B Q4_K_M @32768，8G 独显，
    # 整合包形态副本跑完整首载）：
    #   f16  → 28/34 层在 GPU、23.5 tok/s，连稠密门槛 30 都过不去，首测直接判
    #          "无法可持续"按 32K 记录，新手开局就吃一条 ⚠ 警告
    #   q8_0 → 31/34 层、31.4 tok/s，正常搜到 36K，备注是 ℹ
    # KV 显存砍半还顺带把每 token 的 KV 读取流量减半（稠密模型每 token 扫全部
    # KV，这个折扣直接体现在解码速度上）；质量损失很小。config.example.json
    # 一直都推荐 q8_0，是 DEFAULTS 没跟上 —— 新机器不抄示例就吃亏，不该这样。
    # 改完必须 stop→use 才生效（加载期参数）。
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "idle_unload_sec": 300,         # 模型闲置 N 秒自动卸载释放显存（对齐 Ollama keep_alive；0=关闭）
    # T1b：启动后自动打开浏览器（整合包双击即用）。false = 只打印 URL 不弹
    # 浏览器（服务器场景 / 就是不想被弹窗的人）。
    "auto_open_browser": True,
    # 各模型目标上下文。**默认留空** —— 这是每台机器（显存/量化档不同）各自测出来的
    # 值，写死在代码里对别人毫无意义还容易误导。
    #   填法二选一：
    #     "模型tag": 32768        强制指定
    #     "模型tag": "auto"       无缓存时自动测一次（控制台 measure <模型> 手动测也行）
    # 控制台 `measure <模型>` 会把实测值写进上面的 ctx_measured，之后自动复用。
    "model_ctx_overrides": {},
    # 视觉（VL）模型 → mmproj 投影器文件映射。key 用模型 tag 或底座名均可。
    # **默认留空**：新用户的模型目录里有什么、叫什么，他自己知道。
    # 两条填法（手动写盘时）：
    #   ① 显式映射（推荐，不依赖文件名约定）：
    #      "qwen3.5-4b-ud": "D:\\models\\mmproj-F16.gguf"
    #   ② 什么都不写 —— kaiwu_manager 会兜底扫「模型文件同目录下的 mmproj*.gguf」，
    #      且只在能**自证归属**时才挂（文件名含模型底座名，或该目录只有这一个模型）。
    #      扁平目录里多个模型混放时判定不了，就宁可不挂，让你在卡片上手动点「视觉」。
    # Web UI 的「添加模型文件夹」按钮会自动登记这个映射，不用手改。
    "mmproj_models": {},
    # ---- 投机解码草稿模型（可选）----
    # key 用模型 tag 或底座名；命中就给该模型追加
    # --model-draft <path> --spec-type <type> --spec-draft-n-max <n>。
    # 值可以直接写路径字符串（等价 {"path": ...}），或
    # {"path": "...", "type": "draft-mtp", "n_max": 3}。
    # ⚠️ 代价是草稿常驻显存（gemma-4 MTP 头 Q8_0 ≈ 0.46 GB，还会被算进
    #    显存 fit），8G 卡上会挤掉主模型的一部分层。
    # 2026-10-01 本机实测（gemma-4-26B-A4B IQ3_S + mtp-Q8_0，b10951）：
    #   · 创意长文（散文 500 tok）：接受率 ~43%，30.6/34.6 tok/s —— 与不挂持平
    #   · 结构化短答（列举颜色）：接受率 ~75%，40.1 tok/s —— 明显更快
    #   ⇒ 收益强依赖任务可预测性，不是无脑加速开关。
    "draft_models": {},
    # ---- 挂载开关（Web UI 工具栏的「视觉」「MTP」两个按钮）----
    # attach_vision：配了 mmproj_models 的模型是否真挂视觉塔（--mmproj）
    # attach_draft ：配了 draft_models 的模型是否真挂投机解码草稿
    # ⚠️ 两者都是**加载期参数** ⇒ 改完必须重新加载模型才生效（先 stop 再 use）。
    # 关掉视觉能省约 1.3 GB 显存（实测让纯文本快回约 12%）；
    # 关掉草稿能省约 0.46 GB，但结构化输出会失去那部分加速。
    "attach_vision": True,
    "attach_draft": True,
    # 按模型的挂载偏好（模型卡片上的「视觉」「MTP」标签开关写这里）：
    #   {"模型tag": {"vision": bool, "draft": bool}}
    # 没记录的模型回落到上面两个全局默认值 —— 全局值现在只当"默认"用。
    "attach_overrides": {},
    # ---- Ollama 网关 ----
    # 优先 11434（Ollama 客户端零配置）；被真 Ollama 占用时自动退到备用端口
    "gateway_port": 11434,
    "gateway_fallback_port": 11437,
    # 绑定地址：默认仅本机；设 "0.0.0.0" 对局域网开放
    "gateway_host": "127.0.0.1",
    # API key 门禁：非空时，非本机请求必须带 Authorization: Bearer <key>
    # 或 X-Api-Key: <key>；本机 127.0.0.1 始终免钥匙。空 = 不启用
    "gateway_api_key": "",
    # 客户端未指定生成长度时的默认 max_tokens（思考型模型需要余量）；
    # 客户端显式设置时不会覆盖。上游会自动钳制到剩余上下文内
    "max_tokens_default": 8192,
    # OpenAI 路径（/v1/*）是否也按请求里的 model 名自动切换模型。
    # 默认开：HanaAgent 这类只讲 OpenAI 协议的客户端选了哪个就装哪个。
    # 名字解析不到本地 gguf 时不报错、原样放行（llama-server 本来就忽略 model 名）。
    # 设 false 恢复旧行为：/v1/* 只跟随当前加载的模型，换模型回控制台 use
    "gateway_v1_model_switch": True,
    # 思考链归位：有些本地模型的聊天模板没声明思考标记（如 LFM2.5-8B-A1B，
    # llama-server 识别 thinking=0），于是模型吐出的 <think>…</think> 会整段落进
    # **正文**，客户端只能当普通文本看（还白占上下文）。开启后网关在返回时把
    # 开头的 <think> 挪到 reasoning_content，/v1 与 Ollama 两条路都生效。
    # 对本来就被正确解析的模型（qwen3.5-4b-ud / gpt-oss-20b）是空操作，故默认开
    "normalize_think": True,
    # 工具调用归位（LFM2.5 专用）已于 2026-09-13 随该模型退役一并删除。
    # 原先这里有个 normalize_tools 开关：LFM2.5 把工具调用写在正文里
    # （<|tool_call_start|>[fn(a=1)]<|tool_call_end|>），llama-server 的模板
    # 解析器拆不动会直接 500，网关只好自己归位。现在工具调用一律原样透传上游，
    # 模型必须能被 llama-server 正确解析。代码备份见
    # app/ollama_gateway.py.bak_20260913_lfmretire
    # 本地 Web UI 端口（仅本机回环），被占自动 +1/+2
    "webui_port": 8080,
    # Web UI 发消息时若发现推理服务没在跑（最常见的成因就是 idle_unload_sec
    # 把模型闲置卸载了），自动按 last_model 加载一次，并把进度报到步骤条。
    # 设 false 就退回旧行为：直接报"推理服务连不上"。
    "chat_autoload": True,
    # 每次请求把当前日期时间注入 system（见 app/env_info.py）。默认开。
    # 起因：本地模型权重有训练截止时间，会自称"今天是 2024 年 5 月"，并据此
    # 搜回旧闻自我强化。注入的是**现场取**的本地时间，
    # 不是写死的常量。关掉则 system 只包含用户自定义提示词与工具清单。
    "inject_datetime": True,
    # ---- 让模型读本地文件夹（app/local_fs.py）----
    # 允许读取的根目录白名单。空 = 该功能不可用（模型拿不到任何文件）。
    # 由 Web UI 的「文件夹」抽屉维护，路径必须是已存在的绝对目录。
    "fs_roots": [],
    # 敏感文件**默认就不进文件树、也读不出来**（密钥类：.env / *.pem / id_rsa /
    # config.json / credentials 等，内置名单见 local_fs.SENSITIVE_EXACT/GLOB）。
    # 下面两项用来微调：
    #   fs_ignore_extra：追加要忽略的模式，支持 fnmatch 写法，如 ["*.log", "private*"]
    #   fs_allow_names：强制放行（覆盖内置名单）。**确实要让模型读某个敏感文件时
    #     才加**，比如 ["config.json"] —— 加了就等于把里面的密钥暴露给模型和会话记录
    "fs_ignore_extra": [],
    "fs_allow_names": [],
    # ---- 历史遗留键（**当前无任何代码读取**，为将来接回保留）----
    # 桌面版时代的联网搜索配置。桌面版退役时 bge 重排那一层没有接回来
    # （权重已删），现在是 10 条结果全给模型自己挑，重排基本多余 —— 所以：
    #   rerank_enabled        旧版：是否对搜索结果做 bge 重排
    #   rerank_model          旧版：重排模型名（BAAI/bge-small-zh-v1.5）
    #   search_inject_count   旧版：非 agent 模式下自动注入几条结果到上下文
    # 这三个键**只存在于 data/config.json**（DEFAULTS 里没有他们），删不删都不
    # 影响运行。要做"非 agent 的自动注入模式"或接回 bge 重排时从这里取值。
    # ⚠️ 它们现在**不是开关**：改了不会有任何效果，别被名字误导。
    # ---- Ollama 导入（可选，见 app/ollama_import.py）----
    # 从一个已有的 Ollama 模型目录里把 .gguf 硬链接出来。**默认留空 = 功能关闭**
    #（scan() 对不存在的目录直接返回空列表，不报错）。
    # 想用就把这两项填成你自己的目录，例如：
    #   "ollama_dir": "C:\\Users\\你的用户名\\.ollama\\models",
    #   "import_dir": "D:\\我的模型\\gguf-inbox"
    "ollama_dir": "",
    "import_dir": "",
    # ---- 联网搜索（Tavily）----
    # Key 从 https://app.tavily.com 免费申请。**留空也不影响使用** ——
    # Web UI 首次打开时会主动弹引导，问你要不要填；不想要就点「跳过」。
    "tavily_key": "",
    # ---- 下面这些由控制台 / Web UI 运行时维护，你一般不用手改 ----
    # imported_models: {模型tag: gguf 绝对路径}。Web UI 的「添加模型文件夹」按钮
    #   会自动写；手动登记也可以。list_models() 还会顺带扫描这些路径所在目录，
    #   把目录里新增的 .gguf 也列出来（mmproj*/mtp-* 除外，那俩是伴生文件）。
    "imported_models": {},
    # model_units: {模型单元文件夹: [该文件夹里的模型tag]}。「一模型一文件夹」
    #   约定（2026-10-15，见 app/model_scan.py）的登记表：目录里的 mmproj*/mtp-*
    #   自动归属该目录的模型，同目录同底座名的多个 gguf 算同一模型的量化变体。
    #   由「添加模型文件夹」自动写，不用手改。
    "model_units": {},
    # model_variants: {模型tag: {量化标签: gguf 路径}}。同一模型的多个量化档
    #   （如 Q4_K_M / Q5_K_XL），UI 卡片上出下拉切换；imported_models 里记的是
    #   **当前激活**的那个。同样由「添加模型文件夹」自动维护。
    "model_variants": {},
    # last_model: 上次使用的模型，Web UI 发消息时用它自动加载（config.chat_autoload）
    "last_model": "",
}


# ============================================================ llama-server 定位

# 官方 llama.cpp 从 b10951 起把可执行文件改名成 llama-server.exe（之前是
# llama-server-cuda.exe）；两个都认。
_LLAMA_EXE_NAMES = ("llama-server.exe", "llama-server-cuda.exe")


def _candidate_llama_paths(cfg: dict | None = None) -> list[Path]:
    """按优先级列出 llama-server 可执行文件的候选路径。

    优先级（先命中先用）：
      1. config.llama_server_exe —— 用户显式指定（唯一"权威"来源）
      2. 环境变量 KAIWU_LLAMA_SERVER —— 给打包/脚本场景用，不用改配置文件
      3. 仓库内 bin/ 与根目录 —— 解压即用的整合包布局
      4. ~/.kaiwu/bin*/ —— 用户自己装 llama.cpp 的常用位置
      5. PATH —— 系统级安装

    ⚠️ 每个位置内部**先找新版名 llama-server.exe，找不到才认旧的
    llama-server-cuda.exe**。因为升级时旧 bin 常常原地留着（很常见：
    `~/.kaiwu/bin` 放旧版、`~/.kaiwu/bin-b10951` 放新版，目录名按字典序还正好
    让旧版排前面）。按 exe 名分轮扫而不是按目录一轮扫，避免自动挑到旧版。
    """
    cfg = cfg if cfg is not None else load_config()
    out: list[Path] = []
    seen: set[str] = set()

    def _add(p) -> None:
        try:
            pp = Path(str(p)).expanduser()
        except Exception:
            return
        key = str(pp).lower()
        if key not in seen:
            seen.add(key)
            out.append(pp)

    def _add_user_given(v: str) -> None:
        """用户给的值：像文件名就当文件，不像就当目录再拼两种 exe 名。"""
        v = v.strip().strip('"')
        if not v:
            return
        if v.lower().endswith(".exe"):
            _add(v)
        else:
            for name in _LLAMA_EXE_NAMES:
                _add(Path(v).expanduser() / name)

    _add_user_given(str(cfg.get("llama_server_exe") or ""))
    _add_user_given(os.environ.get("KAIWU_LLAMA_SERVER", ""))

    # 自动发现的目录集合（按优先级）
    dirs: list[Path] = []
    root = app_dir()
    dirs += [root / "bin", root, root / "llama-deps",
             root / "llama-deps" / "bin", root / "llama-server"]
    home = Path.home() / ".kaiwu"
    try:
        dirs += sorted(home.glob("bin*"))
        dirs.append(home)
    except OSError:
        pass

    # 按 exe 名分轮扫：先扫完所有目录的新版名，再扫旧版名
    for name in _LLAMA_EXE_NAMES:
        for d in dirs:
            _add(d / name)

    for name in _LLAMA_EXE_NAMES:
        hit = shutil.which(name)
        if hit:
            _add(hit)
    return out


def resolve_llama_server(cfg: dict | None = None) -> str:
    """找到可用的 llama-server 路径；找不到返回 ""。

    调用方负责把空串翻译成人话错误提示 —— 这里不抛异常，
    免得配置探测阶段就炸掉、控制台连"帮助"都打不开。
    """
    for p in _candidate_llama_paths(cfg):
        try:
            if p.is_file():
                return str(p)
        except OSError:
            continue
    return ""


def llama_server_hint(cfg: dict | None = None) -> str:
    """找不到 llama-server 时给新手的可操作提示（别只丢一句"路径无效"）。"""
    cfg = cfg if cfg is not None else load_config()
    explicit = str(cfg.get("llama_server_exe") or "").strip()
    lines = ["找不到可用的 llama-server 可执行文件。请任选一种方式：",
             "  1) 从 https://github.com/ggml-org/llama.cpp/releases 下载官方 Windows 包，",
             "     解压后把 llama-server.exe 放进程序目录下的 bin\\ 文件夹",
             "     （或直接放在程序根目录）；",
             "  2) 或在程序目录下的 data/config.json 里填 \"llama_server_exe\": \"<完整路径>\"；",
             "  3) 或设环境变量 KAIWU_LLAMA_SERVER 指向它（指文件或目录都行）。"]
    if explicit:
        lines.append(f"注意：配置里当前填的是 {explicit!r}，该文件不存在或不是可执行文件。")
    tried = _candidate_llama_paths(cfg)
    if tried:
        lines.append("已找过：" + "；".join(str(p) for p in tried[:6])
                     + ("……" if len(tried) > 6 else ""))
    return "\n".join(lines)


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    p = config_path()
    if p.exists():
        try:
            user = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(user, dict):
                cfg.update(user)
        except Exception:
            pass  # 配置损坏时回落默认值，绝不让控制台起不来
    return cfg


def save_config(cfg: dict) -> None:
    p = config_path()
    atomic_write_json(p, cfg)


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
