# kaiwu-console

**本地大模型的推理服务 + 对话客户端。** 一个 Python 进程管三件事：把 GGUF 权重
用 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 `llama-server` 跑起来、
提供一个网页版聊天界面、再把 Ollama 协议翻译出去让别的客户端也能接。

全部只监听 `127.0.0.1`，零云端依赖，模型权重留在你自己机器上。

[![Topics](https://img.shields.io/badge/topics-llama--cpp%20%7C%20gguf%20%7C%20local--llm%20%7C%20ollama%20%7C%20moe-blue?style=flat-square)](https://github.com/topics/llama-cpp)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/Avoiking/kaiwu-console/actions/workflows/ci.yml/badge.svg)](https://github.com/Avoiking/kaiwu-console/actions/workflows/ci.yml)

```
┌─ 浏览器 ──────────► Web UI  http://127.0.0.1:8080   （主力入口）
│
├─ Chatbox / ollama CLI ► 网关   http://127.0.0.1:11434（Ollama 协议，零配置）
└─ 任意 OpenAI 客户端 ─► 直连   http://127.0.0.1:11435/v1
                                    ▲
                          llama-server-cuda 进程（同一个）
                          加载你的 .gguf 权重
```

## 它和普通「跑个 llama-server」的差别

**上下文不是手填的，是实测出来的。** 「模型能起来」不等于「能用」——
这版 llama-server 在你指定 ctx 后不会下修上下文，而是把放不下的层赶到 CPU，
于是「能跑」变成「慢跑」。真正的判据是**有多少层还留在 GPU 上**，
`measure` 命令做的就是这件事，测完存下来复用；换机器/换模型会自动作废重测。

**MoE 模型不靠猜，逐档真跑一遍。** 混合专家模型（gemma-4-26B-A4B 之类）
的专家权重可以切在显存和内存之间，切多少直接决定速度。kaiwu 沿 llama-server
的 `--fit-target` 显存预留档位**逐档真加载、实测 tok/s**，自动挑出最快的那一档。
数学只用来剪枝（少测几档），**最终选哪一档只认真测速度**。

**下载即用。** 整合包自带 `llama-server` + 全套 CUDA 依赖，不用装 Python、
不用自己配环境。

---

## 快速开始（Windows）

### 0. 先确认一件事：你的显卡是 NVIDIA

本项目依赖 `llama-server` 的 CUDA 后端。AMD / Intel 显卡请自行编译 CPU 版
`llama-server`（编译方式见 llama.cpp 官方仓库），其余步骤相同。

### 1. 选一条路：下载整合包，或自己跑源码

**路线 A：整合包（推荐新手）**

下载 `kaiwu-console-整合包.zip` 解压即用，**不用装 Python**：

```
kaiwu/
├─ kaiwu.exe          ← 双击就开
├─ _internal/         ← Python 运行时（已打包）
├─ bin/               ← llama-server.exe + 全套 CUDA DLL + OpenSSL DLL
├─ data/              ← 配置（首次启动自动生成）
└─ models/            ← 把你的 GGUF 放这儿，一个模型一个文件夹
```

放好模型 → 双击 `kaiwu.exe` → 浏览器自动打开控制台。

**路线 B：从源码跑**

```bat
pip install -r requirements.txt
python console.py
```

> `requirements.txt` 只有一个 `requests`。Python 3.10+ 均可。

两条路都需要 `llama-server`（路线 A 已经自带）。

### 2. 拿 llama-server（只有走路线 B 才需要）

`llama-server.exe` **不随源码仓库分发**（二进制体积大、平台各异）。从
[llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 下载对应
CUDA 版本的官方 Windows 包，解压后把 `llama-server.exe` 拷到本仓库的 `bin\` 目录下
（没有 `bin` 就自己建一个）。

程序会自动按这个顺序找它，一般不用改配置：

1. `data/config.json` 里的 `llama_server_exe`
2. 环境变量 `KAIWU_LLAMA_SERVER`
3. 仓库的 `bin\` 和根目录
4. `~\.kaiwu\bin*\`
5. `PATH`

> **如果报「进程提前退出」且日志是空的**：多半是缺 OpenSSL DLL
> （`libcrypto-3-x64.dll` / `libssl-3-x64.dll`）。llama.cpp 官方包**不附带**这两个
> 文件，缺了它们 Windows 会直接终止进程、连一行日志都不打。把它们放到
> `llama-server.exe` 同目录下即可 —— 报错信息里也会提示这一点。
> （整合包已经带上了这两个 DLL，走路线 A 不会遇到。）

控制台会自己开两个服务并打印网址：

- **Web UI** <http://127.0.0.1:8080> —— 浏览器打开就能聊
- **推理服务** <http://127.0.0.1:11435> —— 给别的客户端填的地址，浏览器打开也能聊
  （llama.cpp 自带一个简易界面）

### 3. 添加模型

把模型文件夹在网页上点「添加模型文件夹」选过去，完事。不需要手改任何配置文件。

**推荐「一模型一文件夹」摆放**——这是伴生文件（视觉塔 / MTP 头）能自动配对的关键：

```
models/                          ← 选这个目录
├─ gemma-4-26B-A4B-it-UD/        ← 一个子文件夹 = 一个模型
│  ├─ gemma-4-26B-A4B-it-UD-IQ3_S.gguf    主权重
│  ├─ mmproj-gemma-4-26B-BF16.gguf        视觉塔（可选）
│  └─ mtp-gemma-4-26B-A4B-it-Q8_0.gguf    MTP 头（可选）
├─ Qwen3.5-4B-UD/
│  ├─ Qwen3.5-4B-UD-Q5_K_XL.gguf          同底座多个量化 = 变体，
│  └─ Qwen3.5-4B-UD-Q4_K_M.gguf           卡片上出下拉切换
└─ josie-4b-instruct-32k/
   └─ josie-4b-instruct-32k.gguf
```

配对规则（宁缺勿错，不猜）：① 塔/头文件名里含模型底座名 → 挂；
② 文件夹里只有这一个模型 → 挂；③ 还分不清 → 不挂，卡片上给
「视觉 · 待选择」下拉让你手动指定。卡片标签悬停能看到当前配的是哪个
文件、凭什么认上的。

`.gguf` 直接平放在根目录也行（旧布局）：逐个登记、名字用完整文件名，
但多个模型混放时程序不敢给它们分配同目录的塔——别问，问就是怕挂错。

或者在控制台里：

```
kaiwu> list              # 看看扫到哪些模型了
kaiwu> use 1             # 加载第 1 个（真实卸载当前模型、加载新的）
kaiwu> measure 1         # 实测这个模型在你这张卡上能开多大上下文
kaiwu> stop              # 卸载，释放显存
kaiwu> help              # 全部命令
```

### 4. 接别的客户端

- **支持 OpenAI API 的**（Continue / Cursor / Cherry Studio / LobeChat 等）：
  地址填 `http://127.0.0.1:11435/v1`，模型名随意填（llama-server 不校验）。
  换模型回控制台 `use`。
- **支持 Ollama API 的**（Chatbox / Talkio / ollama 命令行）：地址填
  `http://127.0.0.1:11434`。网关会**按请求里的模型名自动切换**，不用手动切。

---

## 几个值得知道的行为

这些和别的工具不一样，第一次用容易撞上：

**同一时刻只有一个模型在显存里。** 切换会切断正在进行的流式回复。等它说完再切，
或者别把同一个 Agent 的 chat / utility / vision 配成不同的本地模型，否则每轮
来回装卸（一次十几秒起）。

**「模型能起来」不等于「能用」。** 这版 llama-server 在你指定 ctx 后不会下修上下文，
而是把放不下的层赶到 CPU（启动日志里的 `load_tensors: offloaded N/M layers`），
于是「能跑」变成「慢跑」。所以真正的判据是**有多少层还留在 GPU 上**，
`measure <模型>` 做的就是这件事：二分找出全层仍在 GPU 的最大 ctx，存下来复用。
> 这类"幽灵块"判据有个坑修过一次：GGUF 里带引擎不认识的块（如 MTP 头）时，
> 那些块永远不会被 offload 却仍计入分母，会让测量一路退化到下限并把错值写进缓存。

**闲置 5 分钟自动卸载**（`idle_unload_sec`）。这是为了腾显存，也对齐 Ollama 的
`keep_alive` 默认行为。所以「闲置一会儿后的第一条消息」一定会触发一次加载，
网页上会显示进度条。

**KV 缓存填 `q8_0`** 能把 KV 显存砍约一半，等于同显存下能开更大的上下文，质量
损失很小。8G 显存上基本都该开 —— `config.example.json` 里默认就是开的。

**思考型模型会给很长的思考链。** 部分模型在 `max_tokens` 太小的时候会把额度全烧在
思考上、正文一个字都不输出（`finish_reason` 显示 `length`）。这种情况调大
max_tokens，或在卡片上关掉思考开关。

---

## 配置

改 `data/config.json`。这个文件**不进版本库**（里面可能有你的联网 API Key）。
需要手动改的时候，照着仓库里的 [`config.example.json`](config.example.json)
改就行 —— 那个文件每个键都带中文说明。

最常动的几项：

| 键 | 作用 |
|---|---|
| `llama_server_exe` | `llama-server.exe` 的完整路径。留空 = 自动发现 |
| `cache_type_k` / `cache_type_v` | KV 缓存量化，`q8_0` 省显存。**加载期参数，改完要 `stop` → `use`** |
| `idle_unload_sec` | 闲置多少秒卸载模型，`0` = 永不卸载 |
| `gateway_api_key` | 想开局域网访问就必填 |
| `tavily_key` | 联网搜索用的 key，<https://app.tavily.com> 免费申请。不填 = 不联网 |

**KV 量化、ctx、mmproj、MTP 草稿都是「加载期参数」** —— 改了必须先 `stop` 再
`use`，浏览器刷新没用。

---

## 联网搜索与读文件（可选）

顶栏「联网」开关打开后，模型会拿到一个 `web_search` 工具（Tavily）。首次打开
网页会主动问你要不要填 key。

「文件夹」抽屉里加了目录之后，模型会拿到一个 `read_file` 工具。这块有几条安全
设计，**加目录前请先看一眼**：

- 密钥类文件（`.env` / `*.pem` / `id_rsa` / `credentials` / `config.json` 等）
  默认**不进文件树、也读不出来**，内置名单在 `app/local_fs.py`。
  备份后缀（`.bak` / `.old` / `~` / `.save`）会先还原再判一次 —— 只挡 `config.json`
  会放过 `config.json.bak_20260909_125446`，而备份里是同一份密钥。
- 读到的文件内容和网页内容都会被包上标记，当作**不受信任的数据**，里面的指令
  不要执行。这能抬门槛，但强注入仍然可能穿透 —— 真正可靠的是别把装密钥的目录加进来。
- `fs_allow_names` 能强制放行敏感文件。加了等于把里面的内容暴露给模型和会话记录。

HTTP 层有三道闸（Web UI 只绑回环）：Host 校验挡 DNS rebinding、Origin 校验挡
跨站 CSRF、文件端点额外要求 `X-Kaiwu` 头逼浏览器预检。**别把 `webui_port`
或 `gateway_host` 改成对外暴露就以为没事。**

---

## 常见问题

**「找不到 llama-server 可执行文件」**
按上面第 1 步放好 `bin\llama-server.exe`，或把完整路径填进
`llama_server_exe`。报错信息里会列出已经找过哪些位置。

**「进程提前退出」但日志里什么都没有**
缺 OpenSSL DLL（`libcrypto-3-x64.dll` / `libssl-3-x64.dll`）。见第 1 步的提示 ——
这类失败 Windows 一个字日志都不打，只能靠「日志为空」这个特征识别，所以程序里
也加了针对性提示。

**Web UI 打不开 / 端口被占用**
默认 8080，被占会自动试 8081、8082。控制台启动时会打印实际用的端口。
改端口用 `webui_port`。

**加载模型很慢甚至超时**
首次加载要编译 GPU 内核（PTX JIT），几十秒到几分钟都正常，取决于驱动和模型。
把 `startup_timeout_sec` 调大（默认 900 秒）。之后就快了。

**显存不够 / 被赶到 CPU 跑**
跑 `measure <模型>` 看真实的可用上下文，比手填数字靠谱。也检查是不是同时挂了
视觉塔和 MTP 草稿 —— 在卡片上关掉它们，各能省约 1.3G / 0.5G 显存。

**模型列表里出现了 `mmproj-*` / `mtp-*`**
那不是能单独加载的模型，是视觉塔和投机解码草稿。当前版本在扫描模型目录时
就会把它们排除（判据是**文件名前缀** `mmproj` / `mtp-` / `mtp.`，见
`app/model_scan.py` 的 `is_sidecar()`）。
所以看到它们，只有两种可能：① 你的 kaiwu 版本旧；② **文件名不是以这三个前缀开头**
（比如 `gemma-4-26B-BF16-mmproj.gguf`）—— 这种请自己改名成 `mmproj-xxx.gguf`。

**能连上但不生成**
先看控制台的 `logs`，日志里会有 `truncated = 1` 之类的原因行。

---

## 项目结构

```
console.py              控制台入口（REPL + chat 子模式）
gateway_headless.py     无控制台起网关（供脚本 / 后台服务用）
app/
  config.py             配置与路径管理（含 llama-server 自动发现）
  kaiwu_manager.py      llama-server 进程管理：启停/切换/模型清单/ctx 实测
  model_scan.py         模型目录扫描：「一模型一文件夹」单元发现 + 量化分组
  webui.py              Web UI（内嵌单页，零外部依赖）
  ollama_gateway.py     Ollama 协议网关 + OpenAI 直通
  ollama_import.py      从 Ollama 模型目录导入 GGUF（硬链接，备用工具）
  local_fs.py           读本地文件夹（路径闸 + 敏感名单 + 防提示注入）
  websearch.py          Tavily 联网搜索（配额闸 + 缓存 + 去重）
  model_caps.py         模型能力探测（读 GGUF 模板判思考旋钮）
  env_info.py           当前时间注入
llama-deps/             llama-server 需要的 OpenSSL DLL
data/                   运行时配置与会话（已 gitignore）
```

---

## 适合谁用

- 手里有 **GGUF 权重**、想在本地跑大模型，不想折腾 llama-server 参数的人
- 用 **Ollama 但觉得默认上下文太小**的人（kaiwu 默认开箱即用的窗口大得多）
- 需要**同时喂给多个客户端**（Chatbox / Cursor / Cherry Studio / 自己的脚本）的人
- 跑 **MoE 混合专家模型**、想知道显存该怎么切的人

**关键词**：本地大模型、GGUF、llama.cpp、Ollama 替代、Ollama GUI、离线部署、
本地推理、上下文窗口、显存优化、MoE、混合专家、模型择优、私有部署、OpenAI 兼容接口。

---

## 致谢

- **[val1813/kaiwu](https://github.com/val1813/kaiwu)** —— 同名的上游项目，
  是一个 Go 写的本地模型部署器（MIT 协议，`Copyright (c) 2026 Kaiwu Contributors`）。
  本项目受它的思路启发 —— 「不靠猜，靠实测来决定上下文和缓存配置」这个取向，
  以及 OpenAI 兼容端点 + 本地优先的定位。**两者是完全独立的项目**：
  本仓库是 Python 实现，0 行 Go 代码，也没有复制上游源码。上游的 MIT 协议
  允许这样做，在此致谢并说明来源。
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** —— `llama-server`
  与本项目的推理后端（MIT）。本仓库不包含它的代码，需自行下载。

## 协议

本项目采用 **Apache License 2.0**，见 [LICENSE](LICENSE)。

注意：Apache 2.0 只适用于本仓库自己的代码。上游 val1813/kaiwu 与 llama.cpp
各自仍是 MIT 协议（见「致谢」），它们都没有被本仓库分发。