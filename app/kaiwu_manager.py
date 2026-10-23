"""kaiwu 推理服务管理器（控制台版）。

直连 llama-server（OpenAI 兼容端点，默认 127.0.0.1:11435）：
- start(model)：确保指定模型就绪；若在跑其它模型则真实切换
  （先停进程卸载权重释放显存，再用新模型重启）
- stop()：无条件停止服务（自己的子进程 + 端口上遗留的进程都杀）
- list_models()：本地可用模型（config.imported_models + 权重目录扫描）

kaiwu CLI 托管路径（kaiwu run / --mode / --fast 等旗标）已随桌面端一起移除，
本模块只负责直连 llama-server。
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import requests

from .config import app_dir, load_config, save_config, resolve_llama_server, llama_server_hint
from . import fit_scan
from . import model_scan

# CPU_Mapped 权重占比阈值：超过就认定"这个卡装不下、速度必然退化"
# （2026-10-16，gemma-4-26B 实测 76% → 1.8 tok/s；全驻显存时为 0）。
# ⚠️ 只对**稠密模型**成立 —— MoE 每 token 只读被激活的专家，CPU_Mapped
# 里躺着的绝大多数专家是冷数据，实测 gemma-4-26B-A4B 76% 在内存仍有
# 18~22 tok/s（2026-10-17 真机复测，见 _test_ctxfit_moe.py）。
_CPU_MAPPED_OK_RATIO = 0.10

# MoE fit 余量的扫描候选序列（MiB），**从保守到激进**（数值越小越激进、
# 放进显存的专家越多）。抽成模块常量是因为两处要用且必须同序：
#   ① measure_ctx 的 scan_fit —— 逐档实测取最快；
#   ② 二分结束后的"最终窗口下复核相邻档"（方案 §6.2）—— 要按它取相邻档。
# ⚠️ 不许把某个值写死成默认：实测最优档随模型变（gemma 落在 256，
# LFM2.5-24B 要更激进才平）。起手 4096 最保守（保证起得来）；正常 5~6 档收工。
_FIT_CANDIDATES = (4096, 2048, 1024, 512, 256, 128, 64)

# fit 档位比较的**跨加载噪声**（tok/s）—— 2026-10-20 晚独立验收报告实测：
# fitt128 与 fitt64 在 114688 下 n_part 与显存完全相同（6795 MiB / 83%，
# 配置上不可能有性能差异），两次加载却测出 32.02 与 33.21，差 **1.19**。
# ⇒ 任何小于 ~1.2 的读数差都分不出档位。两个用途共用这一个量：
#   ① 扫档停止判据（配合 n_part 是否还在下降）；
#   ② 最终窗口复核（§6.2）与"同分取保守档"的容差。
# ⚠️ 别再拿 0.3 当阈值 —— 那是按单次采样噪声 0.11 定的，跨加载噪声是它的 10 倍。
# 这是**兜底默认值**：真正生效的是 config.fit_gain_min_tps（界面上叫「速度差容差」，
# 用户可填）。常量只在配置缺失/非法时用，别在这里调参 —— 改配置去。
_FIT_NOISE_TPS = 1.2

# 扫档前的预热枪数：MoE 专家权重走内存 mmap 时，前 4~5 枪都在喂页缓存
# （实测 18.59→…→32.26，+74%）。正式扫描前先在已就绪的探测负载上补几枪，
# 各档起点才一致（否则"后跑的档天然更热"，会把保守档判得过分差）。
_FIT_WARM_SHOTS = 2

# llama-server 能力探测缓存：{exe 路径: help 输出文本}。--cpu-moe 这类
# 旗标老版本二进制没有，每次测量都跑一遍 --help 要 0.3s，缓存到进程内。
_llama_help_cache: dict = {}


class KaiwuManager:
    # 进程内缓存：{指纹cache_key: 指纹dict}。见 _vram_fingerprint 的说明
    # （省掉每次加载模型时 0.3~0.6s 的 nvidia-smi + --version 子进程）。
    _fp_cache: dict = {}

    # 最近一次测速的原始采样（含被丢掉的预热那一枪），排查用。
    _last_tps_samples: list = []

    def __init__(self, progress=None):
        """progress: callable(str)，状态与 llama-server 日志回调（控制台打印用）。"""
        self._progress = progress or (lambda line: None)
        self._state = "stopped"           # stopped / starting / ready
        self._model = ""                  # 当前目标模型名
        self._cur_ctx = 0                 # 当前加载模型的 ctx-size（0 = 未加载）
        self._error = ""
        self._logbuf: deque[str] = deque(maxlen=2000)
        self._proc: subprocess.Popen | None = None
        self._lock = threading.RLock()    # 串行化 start/stop，防止并发切换互踩
        # 加载守卫：_loading_tid 非 None = 有一个加载/测量会话在进行
        # （含自动测量全程，测量在锁外，见 start() 的说明）。第二个并发
        # 请求在 start() 门口被毫秒级拒绝，而不是阻塞在 _lock 上排队——
        # 排队会让第二个点击挂起数分钟，且测量阶段光靠锁拦不住。
        # 记线程 id 是为了放行 measure_ctx 内部的重入调用（attempt→start，
        # 同一线程），异线程才拒绝。
        self._loading_tid: int | None = None
        self._loading_model = ""
        self._last_active = time.time()   # 最后一次推理活动时间（闲置看门狗用）
        threading.Thread(target=self._idle_watchdog, daemon=True).start()

    # ------------------------------------------------------------ 基础
    def state(self) -> tuple[str, str]:
        with self._lock:
            return self._state, self._model

    def cur_ctx(self) -> int:
        """当前加载模型的 ctx-size；未加载返回 0（Web UI 的上下文用量圈用）。"""
        with self._lock:
            return self._cur_ctx

    @property
    def last_error(self) -> str:
        return self._error

    @property
    def busy_loading(self) -> bool:
        """是否有加载/测量会话在进行（含自动测量全程，见 start() 的加载守卫）。

        /api/use 用它区分两种 start() 返回 False 的情况：
        True  = 被守卫拒绝（另一个模型正在加载 → 回 409，稍后再试）；
        False = 这次是真的加载失败（回 503）。
        """
        return self._loading_tid is not None

    def log_text(self) -> str:
        return "\n".join(self._logbuf)

    def _log(self, line: str) -> None:
        line = line.rstrip()
        self._logbuf.append(line)
        self._progress(line)

    def base_url(self, cfg: dict | None = None) -> str:
        cfg = cfg or load_config()
        return f"http://{cfg.get('host', '127.0.0.1')}:{int(cfg.get('port', 11435))}"

    def probe(self, cfg: dict | None = None, timeout: float = 1.0) -> tuple[bool, list[str]]:
        """返回 (是否就绪, 已加载模型名列表)。"""
        try:
            r = requests.get(f"{self.base_url(cfg)}/v1/models", timeout=timeout)
            if r.status_code == 200:
                data = r.json().get("data", [])
                return True, [m.get("id", "") for m in data]
        except Exception:
            pass
        return False, []

    def status(self, cfg: dict | None = None) -> dict:
        cfg = cfg or load_config()
        ok, served = self.probe(cfg)
        state, model = self.state()
        return {
            "state": state if ok else ("ready" if not ok and state == "ready" else state),
            "endpoint_alive": ok,
            "serving": served,
            "model": model,
            "base_url": self.base_url(cfg),
            "pid": self._proc.pid if self._proc and self._proc.poll() is None else None,
        }

    # ------------------------------------------------------------ 模型清单
    def list_models(self, cfg: dict | None = None) -> list[dict]:
        """本地可用模型：config.imported_models 映射优先，再扫描权重目录发现新 gguf。
        返回 [{"name", "path", "size", "mtime", "imported",
               "variants": {量化标签: 路径}, "active": 当前标签, "folder": 目录}]，
        按名称排序。

        目录约定（2026-10-15「一模型一文件夹」，见 app/model_scan.py）：
          · model_units 登记过的目录 = 模型单元 —— 同目录、同底座名的多个
            gguf 归并成**量化变体**（一张卡片一个模型，变体在 UI 里下拉切换）；
          · 没登记过的目录 = 旧扁平布局 —— 逐个登记，名字用完整文件名。
        """
        cfg = cfg or load_config()
        by_path: dict[str, dict] = {}
        imported = cfg.get("imported_models") or {}
        variants_cfg = cfg.get("model_variants") or {}
        units_reg = cfg.get("model_units") or {}
        unit_dirs = {str(Path(k)).lower() for k in units_reg}

        def _put(name: str, path: Path, imported_flag: bool) -> None:
            try:
                st = path.stat()
            except OSError:
                return
            key = str(path.resolve()).lower()
            if key in by_path:
                if imported_flag:
                    by_path[key]["name"] = name
                    by_path[key]["imported"] = True
                return
            by_path[key] = {
                "name": name, "path": str(path),
                "size": st.st_size, "mtime": st.st_mtime,
                "imported": imported_flag, "variants": {},
            }

        for name, path in imported.items():
            p = Path(path)
            if p.is_file():
                _put(name, p, True)

        roots = {Path(v).parent for v in imported.values() if Path(v).is_file()}
        for k in units_reg:
            kp = Path(k)
            if kp.is_dir():
                roots.add(kp)
        for d in roots:
            is_unit = str(d).lower() in unit_dirs
            try:
                files = sorted(d.glob("*.gguf"))
            except OSError:
                continue
            for f in files:
                # mmproj = 视觉投影器；mtp-* = 投机解码草稿 —— 两者都不是能单独
                # 加载的模型，列进 UI 只会让用户误选后加载失败。
                if model_scan.is_sidecar(f.name):
                    continue
                if str(f.resolve()).lower() in by_path:
                    continue
                base = model_scan.base_stem(f.stem)
                owner = None
                for e in by_path.values():
                    ep = Path(e["path"])
                    if ep.parent != d or model_scan.base_stem(ep.stem) != base:
                        continue
                    # 并成变体（而非新模型）的两种情形：① 目录是登记过的模型
                    # 单元；② 条目本身就是按底座名登记的（单元模式的 imported 项）。
                    # 旧扁平布局里同名不同 quant 的文件仍是各自独立的模型。
                    if is_unit or e["name"].lower() == base.lower():
                        owner = e
                    break
                if owner is not None:
                    lbl = model_scan.quant_label(f.stem) or f.stem
                    owner["variants"][lbl] = str(f)
                    continue
                _put(base if is_unit else f.stem, f, False)

        out = []
        for e in by_path.values():
            vs = dict(e.get("variants") or {})
            cfgv = variants_cfg.get(e["name"])
            if isinstance(cfgv, dict):
                for lbl, p in cfgv.items():
                    if Path(p).is_file():
                        vs[str(lbl)] = str(p)
            # 当前激活的那个必须在变体表里，否则 UI 下拉框会漏掉它
            active = (model_scan.quant_label(Path(e["path"]).stem)
                      or Path(e["path"]).stem)
            if e["path"] not in vs.values():
                vs[active] = e["path"]
            e["variants"] = vs
            e["active"] = active
            e["folder"] = str(Path(e["path"]).parent)
            out.append(e)
        return sorted(out, key=lambda e: e["name"].lower())

    # ------------------------------------------------------------ 启动/切换/停止
    def start(self, model: str, cfg: dict | None = None, timeout: float | None = None,
              ctx_override: int | None = None, force_measure: bool = False,
              moe_cpu: bool = False, extra_args: list[str] | None = None) -> bool:
        """（阻塞）确保 model 就绪。带加载守卫，同一时刻只放行一个加载会话：

        进行中再来请求（典型场景：UI 上点了 A 正在加载，又点 B），在这里
        毫秒级返回 False —— last_error = "正在加载 X，请稍候"，busy_loading
        为 True（调用方据此回 409，提示用户等就绪后再点）。**不阻塞在 _lock
        上排队**：排队会让第二个点击的请求挂起数分钟（最坏 startup_timeout
        _sec），期间前端两个卡片都转圈、顶栏状态还是旧模型，状态全错乱；
        且自动测量发生在拿锁之前，光靠锁连测量阶段都拦不住——两路测量会
        交叉污染同一份 llama-server 日志和 _gpu_offload_info 的解析结果，
        把测错的 ctx 写进 config.ctx_measured。

        守卫检查本身不拿 _lock —— 否则第二个请求照样先阻塞在锁门口，等于没改。
        记线程 id 是为了放行"自己人"：measure_ctx 的 attempt() 会在同一线程
        里重入调用本函数（ctx_override 非 None），必须放行；其它线程一律拒绝。

        moe_cpu：本次加载加 --cpu-moe（MoE 专家权重显式走内存、显存全留给
        KV/计算，见 measure_ctx 的 MoE 段）。自动测量内部传；普通加载不用传
        —— config.moe_offload 记着 "cpu" 的模型@量化档会自动带上。

        extra_args：按原样追加在 llama-server 命令行末尾的额外旗标。自动测量
        的 fit 档位扫描用它传 ["-fitt", "256"]（见 measure_ctx 的 scan_fit）；
        给了 extra_args 就表示"本条的 MoE 旗标由调用方自己决定"，不再叠加
        config.moe_offload 的老记录（否则 --cpu-moe 与 -fitt 会同时上命令行，
        扫描出来的档位全不作数）。
        """
        tid = threading.get_ident()
        if self._loading_tid is not None and self._loading_tid != tid:
            self._error = f"正在加载 {self._loading_model}，请稍候"
            self._log(f"[加载守卫] 拒绝并发加载 {model}"
                      f"（{self._loading_model} 正在加载中）")
            return False
        outer = self._loading_tid is None
        if outer:
            self._loading_tid, self._loading_model = tid, model
        try:
            return self._start_inner(model, cfg, timeout, ctx_override,
                                     force_measure, moe_cpu, extra_args)
        finally:
            if outer:   # 异常路径也要解锁，别把管理器永久卡在"加载中"
                self._loading_tid, self._loading_model = None, ""

    @staticmethod
    def _prefill_args(cfg: dict) -> list[str]:
        """预填批量旗标（-b/-ub）。ubatch > 512 才加，否则返回空（用引擎默认）。

        抽成纯函数是为了可单测：旗标拼错（比如 -b 小于 -ub）llama-server
        会直接拒绝启动，而这种错误只在真实加载时才暴露。
        """
        pub = int(cfg.get("prefill_ubatch") or 0)
        if pub <= 512:
            return []
        pb = int(cfg.get("prefill_batch") or 0)
        if pb < pub:
            pb = pub * 2
        return ["-b", str(pb), "-ub", str(pub)]

    def _start_inner(self, model: str, cfg: dict | None = None, timeout: float | None = None,
                     ctx_override: int | None = None, force_measure: bool = False,
                     moe_cpu: bool = False,
                     extra_args: list[str] | None = None) -> bool:
        """（阻塞）确保 model 就绪。已在跑同一模型直接返回 True；
        在跑其它模型则先卸载再加载；未在跑则直接加载。

        ctx_override：本次加载强制使用的上下文（自动测量内部反复调用本函数用）。
        force_measure：**跳过 _should_measure 的所有判断，无条件先测一次**。
          UI 的「重新测量」用它 —— 换了显卡 / 加了显存 / 换了 KV 量化档之后，
          旧记录可能偏小（保守）或环境指纹压根没变（同一张卡改配置），
          自动判定未必会重测，但用户明确要求了就得照做。
        """
        cfg = cfg or load_config()
        # 自动测量入口：无显式值 + 无测量缓存 + 开了 ctx_auto（或该模型写了 "auto"）
        # → 先量一次，结果写回 config.ctx_measured，随后 _target_ctx 自动采用。
        # 放在拿锁之前：measure_ctx 会反过来调用本函数。
        if ctx_override is None and (force_measure or self._should_measure(cfg, model)):
            self.measure_ctx(model, cfg, timeout)
            cfg = load_config()
        with self._lock:
            self._error = ""
            self._last_active = time.time()
            ctx = int(ctx_override) if ctx_override else self._target_ctx(cfg, model)
            ok, served = self.probe(cfg)
            # 已经在服务目标模型还不够——**上下文必须也是目标值**才认。
            # 自动测量会留下"最后一试"的服务器（很可能是个掉了层的失败档），
            # 那时 _cur_ctx 与目标不符；若不卸掉重载，serve 的就是掉层慢速档，
            # 而 UI 的 cur_ctx 还停在最后一次尝试的 ctx 上，两个数字打架。
            # 2026-10-16 Ornith 实测：测量结论 32768，服务器却留在 45056
            # （31/34 层、3 层掉 CPU）→ 21 t/s，toast 报"上下文 44K"，
            # 卡片却显示"上次测到 32K"（日志证据在 Memorix obs:33）。
            # _cur_ctx == 0 = 非本实例拉起或从未加载过，无从校验，维持原判。
            if (ok and self._model_matches(served, model)
                    and self._cur_ctx in (0, ctx)):
                self._state, self._model = "ready", model
                self._remember_last(cfg, model)
                return True
            if ok:
                if self._model_matches(served, model):
                    self._log(f"重载 {model}：运行中的上下文 {self._cur_ctx}"
                              f" ≠ 目标 {ctx}，卸载重来 …")
                else:
                    self._log(f"切换模型：卸载 {served[0] if served else '当前模型'} …")
                if not self._stop_server_locked(cfg):
                    return False

            model_path = self._resolve_gguf(cfg, model)
            if not model_path:
                self._error = f"找不到模型 {model!r} 的本地 GGUF 文件"
                self._log(self._error)
                return False
            ls = resolve_llama_server(cfg)
            if not ls:
                self._error = llama_server_hint()
                self._log(self._error)
                return False

            args = [ls, "-m", model_path,
                    "--ctx-size", str(ctx),
                    "--port", str(int(cfg.get("port", 11435))),
                    "--host", cfg.get("host", "127.0.0.1"),
                    "--jinja",
                    # b10951+ 默认日志 verbosity=3，把 ggml_cuda_init / system_info /
                    # load_tensors 那批 INFO 行整段压掉了（一次加载只剩 ~24 行）。
                    # 其中 `load_tensors: offloaded N/M layers to GPU` 是 measure 的
                    # 判据（_gpu_offload_info）唯一来源：缺了会把「全层在 GPU」误判成
                    # 「掉层」，测量一路退化到下限还把错值写进 ctx_measured。
                    # -lv 4 的输出量与旧版默认相当（~240 行/次加载），且旧版二进制
                    # 同样接受本参数（回滚后 measure 依旧可用，实测 1621 行仍在
                    # 2000 行日志缓冲内）。
                    "-lv", "4"]
            # CORS 收紧（只在新版二进制上加）：新版默认回显**任意** Origin 且
            # 允许 credentials（实测 evil.example 也拿到 ACAO 头）—— 任何网页
            # 都能借浏览器驱动本地模型。收紧为只回显本机 Origin；非浏览器客户端
            # （CLI / 网关 / 桌面 App）不发 Origin 头，不受影响（实测请求照常 200）。
            # ⚠️ 这个旗标是 b10951 才有的：老二进制报 `error: invalid argument:
            # --cors-origins` 直接秒退（2026-10-20 用 ~/.kaiwu/bin 的老 exe 实测，
            # 码 1）—— 不过能力探测就是"回滚到老二进制"那条路要能走通的先决条件。
            # 老版本没有 CORS 开关（也就不存在"回显任意 Origin"这个洞），跳过即可。
            if self._llama_supports(cfg, "--cors-origins"):
                args += ["--cors-origins", "localhost"]
            else:
                self._log("llama-server 不认 --cors-origins（老版本）→ 跳过 CORS 收紧")
            # MoE 专家分配（2026-10-20 改）：
            # 不再无条件 --cpu-moe（那是把**全部**专家赶进内存、显存空着 3G 不用，
            # 实测白亏 9.7%~22%）。改成传 `-fitt <MiB>` 让 llama.cpp 的 fit
            # 自己按份切分专家权重（每层内部均分，Ollama 用的就是这个机制）。
            # 余量值由自动测量实测扫描选出，存 config.fit_target，**不许写死**。
            #   ① 有 fit_target 记录（= 该档实测过 fit 更优）→ 传 -fitt，
            #      此时**绝不许再传 --cpu-moe**（新档位会不生效）。
            #   ② 没有 fit 记录、但老记录 moe_offload="cpu" → 退回 --cpu-moe
            #      （回滚路径；测量时扫描全失败才会写这条）。
            #   ③ --cpu-moe 的代码路径保留：测量显式要求（moe_cpu=True）还要走。
            if moe_cpu:
                args += ["--cpu-moe"]
                self._log("MoE 混合卸载：--cpu-moe（专家权重走内存，显存留给窗口）")
            elif not extra_args:
                # extra_args 非空 = 调用方（fit 扫描）自带 MoE 旗标，本条不叠加
                fit_target = KaiwuManager._fit_target_of(cfg, model)
                if fit_target and self._llama_supports(cfg, "-fitt"):
                    args += ["-fitt", str(fit_target)]
                    self._log(f"MoE 专家分配：-fitt {fit_target}"
                              f"（fit 按份切分专家权重，显存余量 {fit_target} MiB）")
                elif KaiwuManager._moe_offload_mode(cfg, model) == "cpu":
                    args += ["--cpu-moe"]
                    self._log("MoE 混合卸载：--cpu-moe（专家权重走内存，显存留给窗口）")
            # KV 缓存量化（可选）：q8_0 能把 KV 显存砍一半，换更大窗口 / 更多层上 GPU
            ctk = str(cfg.get("cache_type_k") or "").strip()
            ctv = str(cfg.get("cache_type_v") or "").strip()
            if ctk:
                args += ["--cache-type-k", ctk]
            if ctv:
                args += ["--cache-type-v", ctv]
            # 预填批量：长会话第一轮（冷缓存）要全量预填整段历史，默认
            # 2048/512 下 GPU 利用率很低。实测加大 ubatch 预填 2.26 倍
            # （401→908 tok/s @90K）、解码不变，代价 compute buffer ~+700 MiB
            # —— ctx 测量的算式读的就是这里的真实账目，会自己算进去。
            # 见 config.py 里 prefill_batch / prefill_ubatch 的注释。
            args += KaiwuManager._prefill_args(cfg)
            mmproj = self._resolve_mmproj(cfg, model, model_path)
            if mmproj:
                args += ["--mmproj", mmproj]
            # 投机解码草稿（可选）。注意草稿也常驻显存，且 llama.cpp 在 fit
            # 阶段量不到它（日志有 "failed to measure the memory of the extra
            # model, fitting without it"）⇒ 实际可用余量比 fit 报告的更小。
            draft = self._resolve_draft(cfg, model, model_path)
            if draft:
                dpath, dtype, dnmax = draft
                args += ["--model-draft", dpath,
                         "--spec-type", dtype,
                         "--spec-draft-n-max", str(dnmax)]

            # 测量内部的额外旗标（fit 档位扫描）—— 原样追加在末尾。
            # 参数链必须一路透传 start → _start_inner → 命令行；漏一层就是
            # "扫描跑了但参数没生效"，日志里看不出来（方案 §6.1 坑 1）。
            if extra_args:
                args += list(extra_args)

            self._state = "starting"
            self._model = model
            self._cur_ctx = ctx
            self._log(f"加载模型 {model}（{Path(model_path).name}）")
            self._log("$ " + " ".join(args))
            try:
                self._proc = subprocess.Popen(
                    args,
                    env=self._spawn_env(ls),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            except Exception as e:
                self._error = f"无法启动 llama-server：{e}"
                self._state = "stopped"
                self._log(self._error)
                return False

            threading.Thread(target=self._pump_output, args=(self._proc,), daemon=True).start()

            deadline = time.time() + (timeout or float(cfg.get("startup_timeout_sec", 900)))
            while time.time() < deadline:
                if self._proc.poll() is not None:
                    tail = "\n".join(list(self._logbuf)[-15:])
                    self._error = (f"llama-server 进程提前退出（码 {self._proc.returncode}）\n"
                                   + self._early_exit_hint(tail) + "\n" + tail)
                    self._state = "stopped"
                    self._log(self._error)
                    return False
                ok, served = self.probe(cfg)
                if ok and self._model_matches(served, model):
                    self._state = "ready"
                    self._log(f"就绪：{model} @ {self.base_url(cfg)}")
                    self._remember_last(cfg, model)
                    return True
                time.sleep(0.7)

            self._error = "启动超时：端点长时间未就绪（见上方日志）"
            self._state = "stopped"
            self._log(self._error)
            return False

    def stop(self, cfg: dict | None = None) -> bool:
        cfg = cfg or load_config()
        with self._lock:
            return self._stop_server_locked(cfg)

    # ------------------------------------------------------------ 闲置自动卸载（看门狗）
    def _idle_watchdog(self) -> None:
        """模型就绪后 idle_unload_sec 秒无推理活动 → 自动停止服务释放显存，
        对齐官方 Ollama keep_alive 默认 5 分钟的闲置卸载语义（0 = 关闭）。
        活动信号 = llama-server 端口上的 ESTABLISHED 连接（有请求在途）。"""
        while True:
            time.sleep(10)
            try:
                self._idle_check_once()
            except Exception:
                pass  # 看门狗自身异常绝不能带上宿主进程

    def _idle_check_once(self) -> None:
        cfg = load_config()
        try:
            idle_sec = int(cfg.get("idle_unload_sec") or 0)
        except (TypeError, ValueError):
            idle_sec = 0
        if idle_sec <= 0 or self._state != "ready":
            return
        port = int(cfg.get("port", 11435))
        if self._conns_established(port):
            self._last_active = time.time()
            return
        if time.time() - self._last_active <= idle_sec:
            return
        # 拿不到锁说明 start/stop 正在进行（加载/切换中），本周期跳过
        if not self._lock.acquire(blocking=False):
            return
        try:
            # 双重校验：期间可能刚完成加载，或刚来了新请求（竞态窗口）
            if self._state != "ready":
                return
            if self._conns_established(port):
                self._last_active = time.time()
                return
            # 只停本实例自己拉起的推理进程。端口级 taskkill 带 /T（杀进程树），
            # 有误伤同机上其它服务的风险：2026-09-10 实测看门狗触发后网关(11434)
            # 连同 llama-server(11435) 一起消失，后续请求全部 ConnectionError。
            if self._proc is None:
                self._log("[看门狗] 推理进程非本实例拉起，跳过自动卸载以免误伤")
                self._last_active = time.time()
                return
            self._log(f"[看门狗] 模型闲置超过 {idle_sec}s，自动卸载释放显存（config.idle_unload_sec，0=关闭）")
            self._stop_server_locked(cfg, port_cleanup=False)
        finally:
            self._lock.release()

    @staticmethod
    def _conns_established(port: int) -> bool:
        """llama-server 端口上是否有 ESTABLISHED 连接（= 有推理请求在途）。
        netstat 失败时按"有活动"处理：宁可晚卸载也不误杀生成中的请求。"""
        try:
            r = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True,
                               text=True, timeout=15,
                               encoding="utf-8", errors="replace",  # 中文 Windows GBK 容错
                               creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception:
            return True
        for line in (r.stdout or "").splitlines():
            if "ESTABLISHED" not in line:
                continue
            parts = line.split()
            if len(parts) >= 4 and (parts[1].endswith(f":{port}") or parts[2].endswith(f":{port}")):
                return True
        return False

    # ------------------------------------------------------------ 内部
    def _stop_server_locked(self, cfg: dict, port_cleanup: bool = True) -> bool:
        """杀掉推理服务并等端口释放。自己的子进程优先优雅终止，
        端口上仍有的进程（上次会话遗留）按 PID 强杀。

        port_cleanup=False 时只终止本实例拉起的子进程，不做端口级 taskkill(/T)：
        供闲置看门狗使用，避免按进程树强杀时误伤同机其它服务（如网关 11434）。"""
        self._state = "stopped"
        self._model = ""
        self._cur_ctx = 0
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=8)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if not port_cleanup:
            # 看门狗路径：子进程已优雅终止，直接等端口释放，不按端口强杀
            for _ in range(20):
                ok, _ = self.probe(cfg)
                if not ok:
                    self._log("推理服务已停止（仅本实例进程）")
                    return True
                time.sleep(0.5)
            self._error = "推理进程未能在预期时间内退出"
            self._log(self._error)
            return False
        port = int(cfg.get("port", 11435))
        pids = self._pids_on_port(port)
        for pid in pids:
            if pid == os.getpid():
                continue
            self._log(f"终止端口 {port} 上的进程 PID {pid}")
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True, timeout=15,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            except Exception as e:
                self._log(f"taskkill {pid} 失败：{e}")
        for _ in range(20):
            ok, _ = self.probe(cfg)
            if not ok:
                self._log("推理服务已停止")
                return True
            time.sleep(0.5)
        self._error = f"端口 {port} 释放失败，仍有进程占用"
        self._log(self._error)
        return False

    @staticmethod
    def _pids_on_port(port: int) -> list[int]:
        """netstat 找出 LISTENING 在 port 上的 PID（用于清理非本进程启动的遗留服务）。"""
        try:
            r = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True,
                               text=True, timeout=15,
                               encoding="utf-8", errors="replace",  # 中文 Windows 输出 GBK，容错解码
                               creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception:
            return []
        pids = []
        for line in (r.stdout or "").splitlines():
            if "LISTENING" not in line:
                continue
            parts = line.split()
            if len(parts) >= 5 and parts[1].endswith(f":{port}"):
                try:
                    pids.append(int(parts[4]))
                except ValueError:
                    continue
        return sorted(set(pids))

    @staticmethod
    def _early_exit_hint(tail: str) -> str:
        """进程秒退时先给最可能的原因，别只把最后几行日志甩给用户。

        最常见的是**缺 OpenSSL DLL**：llama.cpp 官方 release 包不附带
        libcrypto-3-x64.dll / libssl-3-x64.dll，Windows 加载器找不到就直接
        终止进程 —— 表现是「日志里什么都没有，进程秒退，退出码非 0」。
        这种情况 stdout 一个字都不会有，所以靠日志内容判不出来，只能按
        「日志为空 + 没有 CUDA 相关报错」来提示。
        """
        low = (tail or "").lower()
        if not tail.strip():
            return ("可能原因：缺少 OpenSSL DLL（llama.cpp 官方包不自带）。\n"
                    "  解法：把 libcrypto-3-x64.dll 和 libssl-3-x64.dll 放到\n"
                    "  llama-server.exe 同目录下，然后重试。\n"
                    "  （若你的包里已经带了，那更可能是 CUDA 运行库缺失 —— 确认 "
                    "nvidia-smi 正常）")
        if "cuda" in low and ("load" in low or "driver" in low or "failed" in low):
            return "可能原因：CUDA 运行环境有问题，确认 nvidia-smi 能正常输出、驱动版本够新。"
        if "no such file" in low or "cannot open" in low or "0xc0000135" in low:
            return "可能原因：DLL 缺失（错误代码 0xC0000135 就是这个意思）。"
        return ""

    def _spawn_env(self, exe: str = "") -> dict | None:
        """构造子进程环境变量。

        为什么要拼 PATH：Windows 加载 DLL 时按 PATH 顺序找，而 llama-server
        依赖 OpenSSL 的 `libcrypto-3-x64.dll` / `libssl-3-x64.dll`。
        **llama.cpp 官方 release 包不带这两个 DLL**（不塞进 llama-server.exe 旁边
        是上游的选择），缺了它们进程会静默秒退、控制台只看到"进程提前退出"。

        依次前置（都存在才加，不存在的目录跳过）：
          1. `llama-deps/`     —— 仓库自带的依赖目录
          2. llama-server.exe **自己所在目录** —— 官方包把 DLL 放 exe 旁边的情形
        """
        env = dict(os.environ)
        extra: list[str] = []

        deps = app_dir() / "llama-deps"
        if deps.is_dir():
            extra.append(str(deps))

        # exe 同目录：官方包把 CUDA/cublas 等一堆 DLL 都放 exe 旁边，
        # 由加载器按同目录优先规则解析；显式前置更稳（尤其配合 gitignore 后的 bin/ 布局）。
        try:
            if exe:
                d = Path(exe).resolve().parent
                if d.is_dir():
                    extra.append(str(d))
        except OSError:
            pass

        if not extra:
            return None
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
        return env

    def _pump_output(self, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stdout:
                self._log(line)
        except Exception:
            pass

    def _remember_last(self, cfg: dict, model: str) -> None:
        """记住「上次使用的模型」。

        ⚠️ 必须**读-改-写磁盘现状**，不能把调用方那份 cfg 整体写回去：
        调用方的快照可能已经过期，整体写回会把别处刚写好的键**静默回滚**
        —— 实测 `measure` 刚算出的 `ctx_measured[模型@量化档]` 就这样被
        start() 覆盖掉了（start 成功 → _remember_last(stale_cfg) → 测量结果消失）。
        `_save_measured` 早就用了「以磁盘现状为基准」的写法，这里对齐。"""
        try:
            if cfg.get("last_model") != model:
                cfg["last_model"] = model
                user = dict(load_config())
                user["last_model"] = model
                save_config(user)
        except Exception:
            pass

    @staticmethod
    def _resolve_gguf(cfg: dict, model: str) -> str:
        """模型名 → 本地 GGUF 路径：完整路径 → imported_models 映射 → 权重目录按名找。"""
        p = Path(model)
        if p.suffix.lower() == ".gguf" and p.is_file():
            return str(p)
        mapped = str((cfg.get("imported_models") or {}).get(model) or "")
        if mapped and Path(mapped).is_file():
            return mapped
        roots = {Path(v).parent for v in (cfg.get("imported_models") or {}).values()
                 if Path(v).is_file()}
        for d in roots:
            cand = d / f"{model}.gguf"
            if cand.is_file():
                return str(cand)
            try:
                hits = [f for f in d.glob("*.gguf")
                        if f.stem.lower() == model.lower()
                        and not f.name.lower().startswith("mmproj")]
                if hits:
                    return str(hits[0])
            except OSError:
                continue
        return ""

    @staticmethod
    def _sidecars_dir(dirpath: str, prefixes: tuple[str, ...]) -> list[Path]:
        """某目录下、指定前缀的伴生文件（视觉塔 / MTP 头）。"""
        d = Path(dirpath or "")
        if not d.is_dir():
            return []
        try:
            return [f for f in sorted(d.glob("*.gguf"))
                    if f.is_file() and f.name.lower().startswith(prefixes)]
        except OSError:
            return []

    @staticmethod
    def _sidecars_in(model_path: str, prefixes: tuple[str, ...]) -> list[Path]:
        """模型文件同目录下、指定前缀的伴生文件（视觉塔 / MTP 头）。

        只看**模型文件自己所在的目录** —— 「一文件夹一模型」约定下，
        同目录的伴生文件天然属于这个模型，不需要猜名字。
        """
        if not model_path:
            return []
        return KaiwuManager._sidecars_dir(str(Path(model_path).parent), prefixes)

    @staticmethod
    def _pick_sidecar(cands: list[Path], want: str,
                      base: str = "", sole_owner: bool = False) -> tuple[str, str]:
        """伴生文件挑法：**能自证归属才挂，否则一律拒绝**。返回 (路径, 依据)。

        ⚠️ 「同目录即同模型」只在**该目录专属于这一个模型**时成立。
        实测过的扁平布局：5 个模型 + 2 个 mmproj + 1 个 mtp
        全混在一起，塔文件名还只有 mmproj-BF16.gguf 这种不带底座名的 ——
        此时"目录里只有一个 mtp 头"绝不等于"这个头属于当前模型"。

        判定顺序（依据名 = 返回值第二项，UI 卡片据此显示配对来源）：
          1. 文件名里含该模型的匹配键 → 自证，直接挂（最可靠，"name"）；
          2. sole_owner（目录里只有这一个模型）→ 整个目录都是它的，挂（"folder"）；
          3. 其余 → 拒绝。留给用户在 Web UI 卡片上手动选，宁缺勿错。
        """
        if not cands:
            return "", ""
        # 匹配键梯度（长优先）：模型 tag 常带 -ud/-it/-A4B 一长串修饰，而塔文件名
        # 通常只写到 gemma-4-26B，用完整名做包含匹配会全部落空。
        for key in model_scan.match_keys(base or want):
            hit = [f for f in cands if key in f.name.lower()]
            if len(hit) == 1:
                return str(hit[0]), "name"
        # 名字自证不了：只有该目录确实专属于此模型时才敢挂
        if sole_owner and len(cands) == 1:
            return str(cands[0]), "folder"
        return "", ""

    @staticmethod
    def _sole_owner_of(model_path: str) -> bool:
        """模型文件所在目录是不是只属于一个模型（按**底座名**分组判定）。

        是 → 这个目录就是它的私有文件夹，「一文件夹一模型」约定成立，
             同目录唯一的伴生文件可以放心挂。
        否 → 扁平共享目录，不猜。

        多量化变体（同底座名、不同 quant 后缀，如 Q4_K_M / Q5_K_XL）算同一个
        模型，不影响判定 —— 2026-10-15 起一个文件夹可以放多个量化档。
        """
        if not model_path:
            return False
        d = Path(model_path).parent
        if not d.is_dir():
            return False
        try:
            mains = [f for f in d.glob("*.gguf")
                     if f.is_file() and not model_scan.is_sidecar(f.name)]
        except OSError:
            return False
        if not mains:
            return False
        return len({model_scan.base_stem(f.stem) for f in mains}) == 1

    @staticmethod
    def _resolve_mmproj(cfg: dict, model: str, model_path: str = "") -> str:
        """视觉投影器路径（只取路径；要「怎么配上的」用 _resolve_mmproj_ex）。"""
        return KaiwuManager._resolve_mmproj_ex(cfg, model, model_path)[0]

    @staticmethod
    def _resolve_mmproj_ex(cfg: dict, model: str, model_path: str = "") -> tuple[str, str]:
        """视觉投影器路径 + 配对依据。→ (路径, source)，source ∈
        "config"（显式映射/登记）/ "name"（文件名自证）/ "folder"（文件夹私有）
        / ""（没配上）。UI 卡片据此告诉用户这个视觉塔是凭什么认上的。"""
        if not KaiwuManager._attach_pref(cfg, model, "vision"):
            return "", ""
        mm = cfg.get("mmproj_models") or {}
        for key in (model, Path_stem(model)):
            v = mm.get(key)
            if v and Path(v).is_file():
                return str(v), "config"
        # 兜底：key 大小写不敏感地作为模型名的**子串**（取最长命中）。
        # 模型 tag 常带量化后缀（gemma-4-26B-A4B-it-UD-IQ3_S / …-qat-UD-Q4_K_XL），
        # 写死全名会让同一底座的不同量化各配一遍。取最长命中是为了避免短 key 乱挂。
        ml = model.lower()
        best_len, best_v = 0, ""
        for k, v in mm.items():
            kl = str(k).lower()
            if kl and kl in ml and len(kl) > best_len:
                sv = str(v)
                if Path(sv).is_file():
                    best_len, best_v = len(kl), sv
        if best_v:
            return best_v, "config"
        # 兜底：扫模型文件同目录的 mmproj*.gguf。
        # 不再要求模型名含 vl/vision（塔文件名常常只有 mmproj-F16.gguf 这种
        # 不带底座名的，按名字卡会把塔全挡掉）；能不能挂交给 _pick_sidecar
        # 按「无歧义才挂」的原则判断。
        base = Path_stem(model)
        hit, how = KaiwuManager._pick_sidecar(
            KaiwuManager._sidecars_in(model_path, ("mmproj",)), model, base,
            sole_owner=KaiwuManager._sole_owner_of(model_path))
        if hit:
            return hit, how
        dirs = {str(Path(model_path).parent if model_path else Path(model).parent)}
        for v in (cfg.get("imported_models") or {}).values():
            p = Path(v)
            if p.parent and p.parent.is_dir():
                dirs.add(str(p.parent))
        for d in dirs:
            # 旧逻辑会"取第一个就挂"，在扁平目录下必然挂错（多个模型共用一批塔）。
            # 统一走 _pick_sidecar 的「无歧义才挂」原则。
            hit, how = KaiwuManager._pick_sidecar(
                KaiwuManager._sidecars_dir(d, ("mmproj",)), model, base)
            if hit:
                return hit, how
        return "", ""

    @staticmethod
    def pending_sidecars(model_path: str, kind: str) -> list[dict]:
        """模型目录里**还没配上归属**的伴生文件（UI「待选择」用）。

        返回 [{"name": 文件名, "path": 完整路径}]。只在模型确实没有可用伴生
        文件时才有意义 —— 告诉用户「这儿有塔，但程序不敢乱挂，你点一下选哪个」。
        """
        if not model_path:
            return []
        d = Path(model_path).parent
        if not d.is_dir():
            return []
        prefixes = ("mmproj",) if kind == "vision" else ("mtp-", "mtp.")
        try:
            return [{"name": f.name, "path": str(f)}
                    for f in sorted(d.glob("*.gguf"))
                    if f.is_file() and f.name.lower().startswith(prefixes)]
        except OSError:
            return []

    @staticmethod
    def _attach_pref(cfg: dict, model: str, kind: str) -> bool:
        """该模型是否挂 kind 能力（"vision" / "draft"）。

        按模型偏好 `cfg.attach_overrides[模型][kind]` 优先；没记录则回落到全局
        默认 `cfg.attach_vision` / `cfg.attach_draft`。模型卡片上的标签开关写前者。
        """
        ov = (cfg.get("attach_overrides") or {}).get(model)
        if isinstance(ov, dict) and kind in ov:
            return bool(ov[kind])
        return bool(cfg.get(f"attach_{kind}", True))

    @staticmethod
    def _resolve_draft(cfg: dict, model: str, model_path: str = ""):
        """投机解码草稿模型（只取草稿；要「怎么配上的」用 _resolve_draft_ex）。"""
        r = KaiwuManager._resolve_draft_ex(cfg, model, model_path)
        return r[0]

    @staticmethod
    def _resolve_draft_ex(cfg: dict, model: str, model_path: str = ""):
        """投机解码草稿 + 配对依据。→ ((path, type, n_max) | None, source)，
        source ∈ "config" / "name" / "folder" / ""，含义同 _resolve_mmproj_ex。

        2026-10-05 新增同目录兜底：原来只有显式映射，用户把 mtp 头和模型放
        同一文件夹也不认。且草稿常驻显存（MTP 头 ≈0.46 GB，8G 卡上是净负担），
        所以扫到也只是「可用」，**要不要真挂仍由 attach_draft 开关决定**。"""
        if not KaiwuManager._attach_pref(cfg, model, "draft"):
            return None, ""
        dm = cfg.get("draft_models") or {}

        def _norm(spec):
            """规范化一条草稿配置 → (path, type, n_max)；不合法返回 None。"""
            if isinstance(spec, str):
                spec = {"path": spec}
            if not isinstance(spec, dict):
                return None
            p = str(spec.get("path") or "")
            if not p or not Path(p).is_file():
                return None
            dtype = str(spec.get("type") or "draft-mtp").strip() or "draft-mtp"
            try:
                dnmax = int(spec.get("n_max") or 3)
            except (TypeError, ValueError):
                dnmax = 3
            return (p, dtype, max(1, dnmax))

        for key in (model, Path_stem(model)):
            spec = dm.get(key)
            if spec:
                r = _norm(spec)
                if r:
                    return r, "config"
        # 兜底：key 大小写不敏感地作为模型名子串（取最长命中），同 mmproj。
        ml = model.lower()
        best_len, best_r = 0, None
        for k, spec in dm.items():
            kl = str(k).lower()
            if kl and kl in ml and len(kl) > best_len:
                r = _norm(spec)
                if r:
                    best_len, best_r = len(kl), r
        if best_r:
            return best_r, "config"
        # 兜底二：模型文件同目录扫 mtp-*.gguf（「一文件夹一模型」约定）。
        hit, how = KaiwuManager._pick_sidecar(
            KaiwuManager._sidecars_in(model_path, ("mtp-", "mtp.")), model,
            Path_stem(model),
            sole_owner=KaiwuManager._sole_owner_of(model_path))
        return ((hit, "draft-mtp", 3), how) if hit else (None, "")

    @staticmethod
    def has_draft(cfg: dict, model: str, model_path: str = "") -> bool:
        """草稿能力 = 能解析到草稿且开关开着（模型卡片上的「MTP」标签用）。"""
        return bool(KaiwuManager._resolve_draft(cfg, model, model_path))

    @staticmethod
    def has_vision(cfg: dict, model: str, model_path: str = "") -> bool:
        """视觉能力 = 能解析到 mmproj。显式映射优先；没有则扫模型文件同目录的
        mmproj*.gguf（不再要求模型名含 vl/vision，见 _resolve_mmproj 兜底段）。"""
        return bool(KaiwuManager._resolve_mmproj(cfg, model, model_path))

    @staticmethod
    def _target_ctx(cfg: dict, model: str) -> int:
        """目标上下文：手填 context_size > 按模型覆盖 > 自动测量缓存 > 32768；
        实际由 llama-server 的显存 fit 自动向下修。

        非数字值（如 "auto"）一律跳过——那是给自动测量用的开关，不是上下文值。"""
        manual = str(cfg.get("context_size") or "").strip()
        if manual.isdigit():
            return int(manual)
        ov = cfg.get("model_ctx_overrides") or {}
        # 先按完整模型名匹配：Path_stem 会把带点号的 tag（qwen3.5-4b-ud）截成
        # qwen3，导致覆盖键永远命不中而落回 32768
        v = ov.get(model) or ov.get(Path_stem(model))
        if str(v or "").strip().isdigit():
            return int(v)
        # 测量缓存按「模型@量化档」取（见 _ctx_read）：不同量化体积不同，
        # 能开的窗口不同，一个模型名一条记录必然张冠李戴
        mv, _key = KaiwuManager._ctx_read(cfg, model)
        if mv > 0:
            return mv
        # 老用法兜底：调用方把完整路径当模型名传时，再按文件名试一次。
        # 平名模型不走这里 —— 那正是 _ctx_read 按档判定后拒绝采信的旧记录。
        stem = Path_stem(model)
        if stem != model:
            stem_v = (cfg.get("ctx_measured") or {}).get(stem)
            if str(stem_v or "").strip().isdigit():
                return int(stem_v)
        return 32768

    @staticmethod
    def _explicit_ctx(cfg: dict, model: str) -> str:
        """显式指定值（手填 context_size > 按模型覆盖），没有则空串。
        返回 "auto" 表示"该模型要求自动测量"。"""
        manual = str(cfg.get("context_size") or "").strip()
        if manual:
            return manual
        ov = cfg.get("model_ctx_overrides") or {}
        v = ov.get(model) or ov.get(Path_stem(model))
        return str(v).strip() if v is not None else ""

    @staticmethod
    def _ctx_active_path(cfg: dict, model: str) -> str:
        """模型当前激活量化档的 GGUF 路径（imported_models → 目录解析）。"""
        p = str((cfg.get("imported_models") or {}).get(model) or "")
        if p and Path(p).is_file():
            return p
        try:
            return str(KaiwuManager._resolve_gguf(cfg, model) or "")
        except Exception:
            return ""

    @staticmethod
    def _ctx_cache_key(cfg: dict, model: str, path: str = "") -> str:
        """上下文测量缓存的键：**模型@量化档**；文件无量化后缀 = 模型名本身。

        为什么按档记：不同量化档的权重体积能差 30%+（IQ3_S vs Q8_0），
        同一窗口大量化会掉层/起不来、小量化白占显存 —— 一个模型名一条
        记录必然张冠李戴（2026-10-15：切量化档要能自动重算上下文）。
        path 可由调用方直接给（UI 轮询路径上卡片自带 active 路径，省一次解析）。
        """
        path = path or KaiwuManager._ctx_active_path(cfg, model)
        lbl = model_scan.quant_label(Path(path).stem) if path else ""
        return f"{model}@{lbl}" if lbl else model

    @staticmethod
    def _ctx_single_variant(cfg: dict, model: str) -> bool:
        """该模型当前是否只有一个量化档（同目录同底座主权重仅一份）。

        旧配置的平名 ctx 记录（2026-10-15 前按模型名记的格式）只有在此时
        才无歧义 —— 多档共存时它可能测自任何一个档，不能采信。
        """
        path = KaiwuManager._ctx_active_path(cfg, model)
        if not path:
            return False
        d = Path(path).parent
        base = model_scan.base_stem(Path(path).stem)
        try:
            n = sum(1 for f in d.glob("*.gguf")
                    if not model_scan.is_sidecar(f.name)
                    and model_scan.base_stem(f.stem) == base)
        except OSError:
            return False
        return n <= 1

    @staticmethod
    def _ctx_read(cfg: dict, model: str, path: str = "",
                  n_variants: int | None = None) -> tuple[int, str]:
        """读该模型【当前量化档】的上下文记录 → (值, 实际命中的键)。

        优先「模型@量化档」；没命中且该模型只有一个量化档时，采信旧平名
        记录（老配置迁移，避免无谓重测）。多档共存时旧记录不可信，返回 0
        —— 调用方（_should_measure）见此会安排自动测量。
        n_variants：调用方已知的量化档个数（UI 轮询路径传卡片上的档数，
        省得 glob 目录）；None = 自己 glob 判断。
        """
        key = KaiwuManager._ctx_cache_key(cfg, model, path)
        measured = cfg.get("ctx_measured") or {}
        v = measured.get(key)
        if v is None and key != model:
            single = ((n_variants is not None and n_variants <= 1)
                      or (n_variants is None
                          and KaiwuManager._ctx_single_variant(cfg, model)))
            if single:
                lv = measured.get(model)
                if lv is not None:          # 真采信了才换键；没找到保持主键，
                    key, v = model, lv      # 那才是将来测量要写进去的键
        s = str(v if v is not None else "").strip()
        return (int(s), key) if s.isdigit() else (0, key)

    def _should_measure(self, cfg: dict, model: str) -> bool:
        """要不要在这个模型加载前做一次自动测量。

        判定顺序：
          · 手写了数字 → 不测（尊重手写值）
          · "auto" → 该模型强制测
          · 当前量化档已有缓存且**环境指纹对得上** → 不测（"测一次记住"）
          · 当前量化档没有缓存（含刚切换过来的新档）→ 测
          · 有缓存但指纹对不上（换显卡/改了 KV 量化/版本变了）→ 重测

        缓存按「模型@量化档」记（_ctx_cache_key）：换量化档就是换一条
        记录，新档没测过就自动测 —— 体积不同，窗口必须分开算。

        ⚠️ 指纹校验是开源场景下的关键防线：ctx_measured 里的数字只对**测它
        那台机器**成立，别人 clone 下来若直接采信，就是拿他的卡跑你算的窗口
        —— 要么爆显存，要么掉层到 CPU 变成 0.3 tok/s。
        """
        ex = KaiwuManager._explicit_ctx(cfg, model)
        if ex.isdigit():
            return False                       # 手写了数字，尊重手写值
        val, key = KaiwuManager._ctx_read(cfg, model)
        if val:
            if ex == "auto" or not self._ctx_cache_valid(cfg, key, model):
                # 有缓存但环境变了（或该模型写了 auto）→ 重新测一次
                return True
            return False                       # 缓存有效，永不再测
        return bool(cfg.get("ctx_auto")) or ex == "auto"

    def granted_ctx(self, cfg: dict | None = None) -> int | None:
        """llama-server 实际授予的上下文（每槽）。读 /props，失败返回 None。"""
        try:
            r = requests.get(f"{self.base_url(cfg)}/props", timeout=5)
            return int(r.json()["default_generation_settings"]["n_ctx"])
        except Exception:
            return None

    def _main_load_section(self) -> tuple[int, int] | None:
        """主权重本次加载的日志区间 [start, end)。

        为什么必须锚主模型：挂了视觉塔/MTP 头时，llama-server 会**分别**加载
        主模型、草稿、塔，各打各的 "loading model tensors" /
        "offloaded N/M layers to GPU" / buffer 尺寸行 —— "最后一条"属于草稿
        或塔（小模型永远全在 GPU），拿它当主模型的账目，就会把"主模型大半
        在内存"误判成"全部在 GPU"（2026-10-16 实测 gemma-4-26B-A4B-IQ3_S
        因此被测出 102400 上下文）。锚点用 `load_model: loading model
        '<主权重路径>'` —— 只有主模型打这行（草稿那行是 "loading draft
        model"，塔那行是 "loaded multimodal model"，都不含该串）。区间覆盖
        print_info（n_ctx_train）→ load_tensors（buffer / offloaded /
        CPU_Mapped）→ llama_context（n_ctx）→ KV cache，到草稿开始加载
        （或缓冲区末尾）为止。
        老版本 llama-server 不打锚点行 → 退回"最后一条 loading model
        tensors"起头的旧行为（可能混入草稿段，但比不解析好）。
        """
        lines = list(self._logbuf)
        anchor = None
        for i in range(len(lines) - 1, -1, -1):
            if "load_model: loading model '" in lines[i]:
                anchor = i
                break
        if anchor is None:
            for j in range(len(lines) - 1, -1, -1):
                if "loading model tensors" in lines[j]:
                    return j, len(lines)
            return None
        end = len(lines)
        for j in range(anchor + 1, len(lines)):
            if "loading draft model" in lines[j]:
                end = j
                break
        return anchor, end

    def _gpu_offload_info(self) -> tuple[int, int] | None:
        """**主权重**本次加载的 `offloaded N/M layers to GPU` → (N, M)。

        这版 llama-server 在用户指定 ctx 后 **不会下修上下文**，放不下的
        部分不报错：要么整层赶下 CPU，要么保持层数不变、把权重切块 mmap
        到内存按需流入（CPU_Mapped 行，判据见 _weights_fully_resident）。
        所以"能不能起来"不是判据，"层还在 GPU 上多少 + 权重 residency"
        才是速度判据。
        ⚠️ 只认**主模型段**里第一条（见 _main_load_section）—— 草稿/塔
        也各打一条自己的 "offloaded 5/5 layers to GPU"，取错就是把
        "主模型半掉在 CPU" 读成 "全部在 GPU"。
        ⚠️ **分母里有"幽灵块"**：GGUF 里带了引擎不认识的块时（实测 qwen35 系的
        `blk.32.nextn.*` MTP 头），llama.cpp 会逐个打
        `W model has unused tensor blk.32.xxx -- ignoring` 把它丢掉 ——
        这些块**永远不会被 offload，却仍计入分母 M**。于是 `N == M` 恒为假：
        measure 的收敛条件永远不成立，一路退化到下限 ctx_auto_min 并把那个
        错值写进 ctx_measured。判据：**只有当缺口 (M-N) 恰好等于被忽略的块数时，
        才认定「掉的那几层」就是幽灵块、实际全层仍在 GPU**；缺口对不上就原样返回，
        绝不放宽真掉层的判定。"""
        sec = self._main_load_section()
        if not sec:
            return None
        start, end = sec
        lines = list(self._logbuf)
        pat = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
        hit = None
        for i in range(start, end):
            m = pat.search(lines[i])
            if m:
                hit = (i, int(m.group(1)), int(m.group(2)))
                break
        if hit is None:
            return None
        i, on_gpu, total = hit
        unused = {mm.group(1)
                  for ln in lines[start:i]
                  for mm in [re.search(r"unused tensor (blk\.\d+)\.", ln)] if mm}
        if unused and total - on_gpu == len(unused):
            total = on_gpu
        return on_gpu, total

    @staticmethod
    def _weights_fully_resident(acct: dict | None) -> bool:
        """权重是否全驻显存 —— 2026-10-16 起与"层数在 GPU"并列的判据。

        "offloaded N/M layers" 只说明**层数**归属，不说明 residency：
        b10951 放不下时可能保持层数不变、把权重切块 mmap 到内存按需流入
        （gemma-4-26B 实测：31/31 但 CPU_Mapped 10745 MiB，1.8 tok/s）。

        两条规则（第二条来自 WB seq 25 的 Ornith 实证，没有它带 nextn
        幽灵块的 qwen35 模型每档都会被冤杀成掉层）：
          ① **幽灵块豁免**：层缺口 >0 且恰好等于被 --ignoring 丢弃的块数
             → 掉下来的全是永不计算的幽灵块，真实层全在 GPU；此时
             CPU_Mapped 里只剩冷数据（幽灵张量 + 嵌入表），不拖速度——
             Ornith 实测 33/34 + CPU_Mapped 683 MiB（占比 12.8%）仍 43.28 t/s。
             （缺口为 0 时绝不豁免：gemma 就是 31/31 缺口 0 的切块形态。）
          ② **切块 mmap 占比**：其余形态看 CPU_Mapped 里的**热**权重占比
             （分子剔除可精确求和的幽灵张量字节，见 _parse_vram_log 的
             ignored_mib）是否超过 _CPU_MAPPED_OK_RATIO。
        读不到账目（老版本 llama-server）→ 只按层数判，保持旧行为。
        """
        if not acct:
            return True
        gap = int(acct.get("layer_gap") or 0)
        ignored = int(acct.get("ignored_count") or 0)
        if gap > 0 and gap == ignored:
            return True
        cpu_m = float(acct.get("cpu_mapped_mib") or 0.0)
        gpu_w = float(acct.get("model_mib") or 0.0)
        if cpu_m <= 0:
            return True
        hot = max(cpu_m - float(acct.get("ignored_mib") or 0.0), 0.0)
        return hot <= _CPU_MAPPED_OK_RATIO * (cpu_m + gpu_w)

    # ------------------------------------------------- MoE 混合卸载（小显存大 MoE）
    #
    # 2026-10-17：gemma-4-26B-A4B-IQ3_S 在 8G 卡上的真相（真机复测，
    # 完整服务配置 = 主模型 + MTP 草稿 + 视觉塔 + KV q8_0）：
    #
    #   默认卸载 @98304：fit 自动把 2404 MiB 权重留显存、其余 CPU_Mapped
    #                    → 12.2 tok/s
    #   --cpu-moe @98304：专家全走内存 mmap，显存只驻注意力/嵌入 1964 MiB
    #                    → 18.7 tok/s（更快且更省显存）
    #   --cpu-moe @131072 → 16.3 tok/s（VRAM 7172/8151）
    #
    # 即「权重超出显存」对 MoE 根本不是病态：每 token 只读 8/128 个激活
    # 专家，CPU_Mapped 里绝大多数专家是冷数据，页缓存命中时流入很快。
    # 旧逻辑把 gemma 冤判成"装不下、2~3 tok/s、换更大显卡"完全错误 —
    # kaiwu 控制台的存在意义就是让小显存用户玩上大 MoE。
    #
    # 判据因此分家：
    #   · 稠密模型：权重必须全驻显存（每 token 读全部权重，mmap 一半就慢一半）
    #   · MoE：层数在 GPU + **实测 tok/s 达门槛**即算可持续；优先试
    #     --cpu-moe（显存全留给 KV/计算，窗口能翻几倍），由实测速度定夺。

    @staticmethod
    def _is_moe(acct: dict | None) -> bool:
        """账目是否来自 MoE 模型（llama-server print_info: n_expert > 0）。

        只认引擎自报的原始信号；"引擎没开口"的情况见 _detect_moe。
        """
        try:
            return int((acct or {}).get("n_expert") or 0) > 0
        except (TypeError, ValueError):
            return False

    # 名称里的 MoE 标记（_detect_moe 的兜底，仅引擎未自报时用）。
    # 高精度优先：漏判按稠密处理只是窗口保守 + 备注如实（用户可手动调）；
    # 错判（稠密进 MoE 桶）会按 MoE 的 15 门槛放行半速大窗口 —— 那是
    # 用户明令禁止的"拿半速糊弄"。所以只收高精度标记，不收 "expert"
    # 这类会误伤 "MathExpert-7B"（稠密微调）的宽松词。
    _MOE_NAME_RE = re.compile(
        r"(?:^|[-_ .])moe(?:$|[-_ .])"      # 独立 moe 字样：Qwen3-MoE / foo_moe
        r"|mixtral"                          # Mixtral 全家皆 MoE
        r"|(?:^|[-_ .])a\d{1,3}b(?:$|[-_ .])",  # 激活参数写法：-A4B / _a3b / -A37B
        re.IGNORECASE)

    @classmethod
    def _looks_moe_by_name(cls, name: str) -> bool:
        """模型名/路径里有没有高精度 MoE 标记（兜底判据，见 _detect_moe）。"""
        return bool(cls._MOE_NAME_RE.search(str(name or "")))

    @staticmethod
    def _detect_moe(acct: dict | None, model: str = "") -> bool:
        """判断模型是不是 MoE。**三级判据，对任何新模型都成立**：

        1. **引擎自报**（权威）：加载日志 print_info 段的 n_expert > 0。
           llama.cpp 读的是 GGUF 里的真实架构 —— 与模型叫什么名、谁训的、
           白名单里有没有，全都无关。所以未来加任何新模型都不用改代码。
           print_info 段在、但没有 n_expert 行 = 引擎明确说"稠密"。
        2. **名称兜底**（仅引擎没开口时）：print_info 段整段缺失
           （未来 llama.cpp 改字段名/改日志格式）时，按名字里的高精度
           MoE 标记认（moe / mixtral / -A4B 激活参数写法）。
        3. **都没有 → 稠密**：保守方向。漏判的代价是窗口偏保守 + 备注
           如实（可手动调大）；错判的代价是稠密模型按 MoE 门槛放行
           半速大窗口 —— 用户明令禁止。

        锚定说明：acct 必须来自 _parse_vram_log（已锚定主模型段）——
        挂了视觉塔/MTP 头时，草稿段自己也会打 n_expert=0，取错会把
        MoE 判成稠密（2026-10-16 实测踩过）。
        """
        try:
            if int((acct or {}).get("n_expert") or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
        if (acct or {}).get("print_info_seen"):
            return False          # 引擎明确没提 n_expert = 稠密
        return KaiwuManager._looks_moe_by_name(model)

    def _llama_supports(self, cfg: dict, flag: str) -> bool:
        """当前 llama-server 二进制是否支持某旗标（--help 探测，进程内缓存）。

        --cpu-moe / -fitt / --cors-origins 都是较新的旗标，用户手里的 llama.cpp
        版本不一；老二进制传了会直接报错秒退（`unrecognized option` 或
        `invalid argument`）。用之前先问一次，别拿真实加载去试错。

        ⚠️ 探测**必须带 `_spawn_env` 的 PATH**：llama.cpp 官方包不带 OpenSSL
        DLL，缺 DLL 时进程静默秒退、help 一个字都不输出 —— 不带 PATH 就会把
        **支持**的二进制一律判成"不支持"（2026-10-20 实测：`~/.kaiwu/bin` 里的
        老 `llama-server-cuda.exe` 明明支持 `-fitt`/`-lv`，探测却返回 False，
        因为它的 DLL 在 llama-deps 里）。
        ⚠️ 探到空**不写缓存**：一次偶发失败（杀软拦一下 / 句柄占用）不该把结论
        钉死整个进程生命周期；下次调用重探一次（~0.3s）。
        """
        exe = resolve_llama_server(cfg)
        if not exe:
            return False
        text = _llama_help_cache.get(exe)
        if not text:
            try:
                r = subprocess.run([exe, "--help"], capture_output=True,
                                   text=True, timeout=15,
                                   encoding="utf-8", errors="replace",
                                   env=self._spawn_env(exe),
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                text = (r.stdout or "") + (r.stderr or "")
            except Exception:
                text = ""
            if text:
                _llama_help_cache[exe] = text
        return flag in (text or "")

    def _measure_tps(self, cfg: dict, tokens: int | None = None,
                     timeout: float | None = None, rounds: int | None = None,
                     stat: str = "max", drop: int | None = None) -> float | None:
        """**真实测速**：对就绪中的服务发几个小 completion，读引擎回报的
        timings.predicted_per_second（纯解码速度，不含 prompt 处理）。

        为什么必须真测：日志账目只能说"层数在哪、权重在哪"，说不出速度 —
        2026-10-17 实测同是 31/31 层全在 GPU，default 卸载 12.2 tok/s、
        --cpu-moe 18.7 tok/s，而 --cpu-moe@196608 挂草稿时只有 1.18 tok/s
        （VRAM 7200/8151 贴顶，KV/compute 互相挤）。判"可持续"只能靠跑。

        **为什么不能只跑一次**（2026-10-17 实证，这一段是血的教训）：
        单次采样噪声大到能相差十倍 —— 同一张 8G 卡、同一个 gemma IQ3_S、
        同一个 163840 --cpu-moe，自动化测量读到 **2.04 tok/s**（当场判这一档
        "不可持续"，二分掉到 126976），而几分钟后手动全新加载同一档第一次
        请求就是 **19.31 tok/s**，后续稳定在 22~26。1 次请求的读数既要承担

        两条系统性偏差，都用采样化解：
          ① 前几枪自带预热：CUDA graph 捕获、采样器初始化、**10.7G 专家权重
             mmap 首次触页缺页** —— 实测首次比后续低 25%~67%。MoE 走内存
             mmap 时页缓存要逐渐喂热，**前 4~5 枪都还在爬坡**（2026-10-20 晚
             独立复测，连打 32768+fitt512：29.90 → 32.88 → 33.65 → 34.34 →
             34.46 → 34.60 → 34.56；94208+fitt256 更夸张 18.59 → … → 32.26，
             +74%）。所以**默认丢前 2 枪**（ctx_tps_drop），而不是只丢第 1 枪
             —— 只丢 1 枪等于拿冷态读数当结论（本机早先那批 21.96 / 23.60 /
             20.02 / 22.83 全是这么来的，已作废）。
          ② 剩余几次被瞬时争用拉低是**单向**的：维持 VRAM 需求的资本开销
             只会让它变慢，不会让它变快。真的在换页的配置每一次都慢，取样
             再多次也救不回来；偶发被拖的则至少有一次反映真实能力。所以
             判"能不能跑"时剩下的取**最大值**。

        采样值留在 self._last_tps_samples，日志和排查能看见全过程（不必只看结论数）。

        stat —— **两个统计量的分工，别"统一"掉**（2026-10-20 方案 §改动 4）：
          · "max"（默认）：判"这档能不能跑"（窗口二分的门槛）。争用只会单向
            把速度往下拉、不会往上抬，max 才反映真实能力。
          · "mean"：比较**档位优劣**（fit 扫描）。比的是相对高低，max 会被
            偶发高值带偏，于是把更好的档位判低。
        drop —— 丢前几枪（默认 ctx_tps_drop，5 枪以前按 1 枪兜底）。7 枪丢 2
        正好是独立验收报告建议的"丢前 2、取后 5"。

        失败（请求超时/服务异常/老版本没这个字段）返回 None —— 调用方按
        "测不到"处理，绝不猜一个数填进去。
        """
        n = int(tokens or int(cfg.get("ctx_tps_tokens") or 48))
        to = float(timeout or float(cfg.get("ctx_tps_timeout_sec") or 120))
        r = int(rounds or int(cfg.get("ctx_tps_rounds") or 3))
        dropped = int(drop if drop is not None
                      else (cfg.get("ctx_tps_drop") if cfg.get("ctx_tps_drop") is not None
                            else (2 if r >= 5 else 1)))
        self._last_tps_samples = []
        for _ in range(max(1, r)):
            try:
                resp = requests.post(f"{self.base_url(cfg)}/completion",
                                     json={"prompt": "用一句话介绍你自己。",
                                           "n_predict": n, "temperature": 0.7,
                                           "cache_prompt": False},
                                     timeout=to)
                tps = (resp.json().get("timings") or {}).get("predicted_per_second")
                if tps:
                    self._last_tps_samples.append(float(tps))
            except Exception:
                pass
        if not self._last_tps_samples:
            return None
        tail = self._last_tps_samples[max(int(dropped), 0):] or self._last_tps_samples
        if len(tail) == 1:
            return tail[0]
        return (sum(tail) / len(tail)) if stat == "mean" else max(tail)

    @staticmethod
    def _moe_offload_mode(cfg: dict, model: str) -> str:
        """该模型【当前量化档】记录在案的 MoE 卸载模式（"cpu" / "" = 没记）。

        测量选定 --cpu-moe 后写进 config.moe_offload，以后每次普通加载
        都带上同一个旗标 —— 否则测出 192K 后下次用默认卸载重跑，fit 行为
        不同，测得的窗口未必还成立。
        """
        mode = (cfg.get("moe_offload") or {}).get(
            KaiwuManager._ctx_cache_key(cfg, model))
        return "cpu" if str(mode) == "cpu" else ""

    @staticmethod
    def _fit_target_of(cfg: dict, model: str) -> int | None:
        """该模型【当前量化档】记录在案的 fit 余量（MiB）；没记录返回 None。

        与 _moe_offload_mode 同源同键（_ctx_cache_key）。设计要点：**"有
        fit_target 记录"就是"这模型是 MoE 且实测过 fit 按份切分更优"的判据**
        —— 加载时手里没有显存账目，没法判断 MoE，用这张表代替即可，不用
        额外探测。稠密模型永远不会有这条记录 ⇒ 行为不变。

        只在 _start_inner 里决定参数时说"有没有记录"，不拿它判断"值合不合理"
        —— 值本身由测量扫出来的实测最优档，改它等于改结论。
        """
        v = (cfg.get("fit_target") or {}).get(
            KaiwuManager._ctx_cache_key(cfg, model))
        try:
            return int(v) if v else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _fit_gain_min(cfg: dict) -> float:
        """**速度差容差**（tok/s）：两档测速差小于它，就当作一样快。

        界面上叫「速度差容差」，不叫"噪声"——后者是测量术语，新手看不懂。
        它同时用于三处：扫档停止、最终窗口复核、几个"一样快"的档里选更省
        显存的。

        值来自 config.fit_gain_min_tps（用户可填），缺省/非法时用模块常量
        _FIT_NOISE_TPS（本机实测 1.19 的跨加载差值）。

        为什么要留可配置：同一台机器同一配置重复加载，读数本身会差一点；
        换机器/换模型后这个差值会变（快的小模型绝对值大、大显存机器层数多
        波动也大）。写死一个数就会在别人机器上判错档。

        合法范围 (0, 50]：<=0 等于"任何差异都当真"（会被抖动牵着走），
        过大等于"永远只测一档"（放弃扫描）。超范围按默认值处理，并留一行
        日志——静默改用默认值会让用户以为自己填的值生效了。
        """
        raw = cfg.get("fit_gain_min_tps")
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return _FIT_NOISE_TPS
        if not (0 < v <= 50):
            return _FIT_NOISE_TPS
        return v

    @staticmethod
    def _note_moe_hybrid(acct: dict, tps: float | None, ctx: int,
                         fit: int | None = None) -> str:
        """MoE 混合卸载成功时的卡片备注（ℹ 前缀 = 信息，不是警告）。

        写清楚三件事：专家权重怎么摆的（fit 按份切分 / --cpu-moe 全下放）、
        显存实际驻了多少、实测多少 tok/s。用户看到 10G 权重跑在 8G 卡上时
        有权知道真相，而不是一句"权重超出显存"。

        fit 非空 = 走的是 fit 按份切分（2026-10-20 起的默认形态，传
        -fitt <fit>）；空 = 退回 --cpu-moe（专家全部下放内存）。两句话术
        必须分开写 —— 把 fit 切分说成"专家权重走内存"就把结论说错了
        （方案 §改动 6）。
        """
        n_exp = int(acct.get("n_expert") or 0)
        model_mib = float(acct.get("model_mib") or 0.0)
        n_part = int(acct.get("n_part") or 0)
        tot_l = int(acct.get("total_layers") or 0)
        layers = acct.get("on_gpu")
        total = acct.get("total_layers")
        k = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
        tps_s = f"{tps:.1f} tok/s" if tps else "速度未测到"
        lay_s = (f"{layers}/{total} 层全在 GPU"
                 if layers and total and layers == total else "层数以日志为准")
        if fit:
            # n_part = fit 切出的份数；总层数 - n_part = 有整层专家权重在显存
            if n_part and tot_l:
                split_s = (f"fit 按份切分专家权重（n_part={n_part}，"
                           f"{max(tot_l - n_part, 0)} 层在显存，"
                           f"余量 {fit} MiB）")
            else:
                split_s = f"fit 按份切分专家权重（余量 {fit} MiB）"
        else:
            split_s = (f"{n_exp} 个专家权重在内存 mmap 按需流入"
                       f"（每 token 只读激活的几个）")
        return (f"ℹ MoE 混合卸载：{split_s}"
                f"，显存驻注意力/嵌入 {model_mib / 1024:.1f}G；实测 {tps_s} @ {k}"
                f"（{lay_s}）。小显存跑大 MoE 的正常形态")

    @staticmethod
    def _note_moe_degraded(acct: dict, tps: float | None, ctx: int,
                           floor: float) -> str:
        """MoE 混合卸载也摸不到速度门槛时的备注（实测数据 + 诚实归因）。

        建议只给"榨干硬件"方向的（换更小量化档 / 手动调窗口）——
        kaiwu 的存在就是让小显存用户把硬件榨到极致，不劝人升级机器
        （用户 2026-10-17 守则）。
        """
        k = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
        tps_s = f"{tps:.1f} tok/s" if tps else "速度未测到"
        return (f"⚠ MoE 混合卸载实测也只有 {tps_s}（低于 {floor:.0f} tok/s "
                f"门槛，内存带宽是瓶颈），已按 {k} 记录；"
                f"可换更小量化档或手动调大窗口再试")

    @staticmethod
    def _note_dense_degraded(acct: dict, tps: float | None, ctx: int,
                             total_vram_mib: int, floor: float = 0.0) -> str:
        """稠密模型装不下时的备注：实测速度替换旧文案里编造的"2~3 tok/s"。

        稠密模型每 token 要读**全部**权重，大半在内存 mmap 时速度就是会
        塌 —— 这是物理，不是配置错误。建议保留，但数字必须是实测的。

        两条硬规矩（2026-10-17，都是被打脸之后立的）：
          ① **不许编造因果**：旧文案写"权重 X 超出显存 Y"，可 Ornith 那次
             X=5.2G、Y=8G —— 明明装得下，句子自己在说谎。现在只有真的比
             显存大的时候才这么说，装得下就如实写"没全驻"。
          ② **不许劝人换硬件**：实测 31.5 tok/s 的档位被告知"换更小量化档
             或更大显存卡才能改观"，是在用一句猜的结论推翻跑出来的证据。
             真正摸不到速度门槛的时候才给建议，且建议要带实测数。
        """
        w_mib = (float(acct.get("model_mib") or 0.0)
                 + float(acct.get("cpu_mapped_mib") or 0.0))
        cpu_mib = float(acct.get("cpu_mapped_mib") or 0.0)
        pct = (cpu_mib / w_mib * 100) if w_mib > 0 else 0.0
        tps_s = f"实测 {tps:.1f} tok/s" if tps else "速度未测到"
        k = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
        floor_s = f"{float(floor or 5.0):.0f} tok/s"
        # 权重真的比显存大时才说"超出"；否则是"没全驻"（有 KV/旁路的账要算）
        head = (f"权重 {w_mib / 1024:.1f}G 超出显存 "
                f"{total_vram_mib / 1024:.0f}G"
                if w_mib > float(total_vram_mib or 0) > 0
                else f"权重 {w_mib / 1024:.1f}G 没能全驻"
                     f"（显存 {total_vram_mib / 1024:.0f}G 里还要留 KV 和计算缓冲）")
        why = "稠密模型每 token 都要读全部权重"
        # 速度其实不错 → 不给"换硬件"的建议，只把数字摆出来
        if tps and tps >= float(floor or 5.0):
            return (f"⚠ {head}，{pct:.0f}% 在内存 mmap，{tps_s}（{why}）；"
                    f"这个档位掉出了全部请求的判据，记录值偏保守 —— "
                    f"实际用起来通常没问题，可手动调大窗口试试")
        # 真降级：⚠ + 实测数 + 门槛数，建议只给"榨干硬件"方向的
        # （换更小量化档 / 手动调窗口），**不许写"更大显存卡"** ——
        # kaiwu 的存在就是让小显存用户把硬件榨到极致（用户 2026-10-17 守则）
        return (f"⚠ {head}，{pct:.0f}% 在内存 mmap，{tps_s}（{why}，"
                f"低于 {floor_s} 门槛）；已按 {k} 记录，"
                f"可换更小量化档或手动调大窗口再试")

    @staticmethod
    def _note_dense_partial(acct: dict, tps: float | None, ctx: int,
                            floor: float) -> str:
        """稠密模型"有一部分权重在内存 mmap 但实测够快"时的卡片备注（ℹ）。

        2026-10-17 实证：Ornith-1.5-9B Q5_K_XL 在 8G 卡上 31/34 层、
        959 MiB / 5.2G = 18% 权重走内存，实测 31.5 tok/s —— 比不少全驻模型
        还快。旧逻辑照着 10% 的比例阈值判它"病态"，给它扣了个 ⚠ 还劝人换
        显卡。比例是猜的，速度是跑出来的，这里按后者说话。
        """
        w_mib = (float(acct.get("model_mib") or 0.0)
                 + float(acct.get("cpu_mapped_mib") or 0.0))
        cpu_mib = float(acct.get("cpu_mapped_mib") or 0.0)
        pct = (cpu_mib / w_mib * 100) if w_mib > 0 else 0.0
        k = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
        tps_s = f"{tps:.1f} tok/s" if tps else "速度未测到"
        layers = acct.get("on_gpu")
        total = acct.get("total_layers")
        if layers and total:
            lay_s = (f"{layers}/{total} 层全在 GPU" if layers == total
                     else f"{layers}/{total} 层在 GPU，{int(total) - int(layers)} 层在 CPU")
        else:
            lay_s = "层数以日志为准"
        return (f"ℹ 稠密混合：{pct:.0f}% 权重在内存 mmap（这部分每 token 要经"
                f" PCIe 读一遍），显存驻 {float(acct.get('model_mib') or 0.0) / 1024:.1f}G；"
                f"实测 {tps_s} @ {k}（{lay_s}，判定门槛 {floor:.0f} tok/s）。"
                f"占比超过保守阈值，但实测够快 —— 以实测为准"
                f"（这个数是近空上下文测的，上下文填满后还会再慢一些）")

    # ------------------------------------------------- 显存解析（单次加载定 ctx）
    #
    # 为什么要这一层：旧的 measure_ctx 是二分搜索，一个模型最多 10+ 次真实加载。
    # 在 8G 卡上 gemma 每次加载十几秒，跑完一个模型要好几分钟 —— 慢到没人愿意
    # 在日常使用里开着，于是 ctx_auto 长期是 false，缓存也就永远是空的。
    #
    # 但 llama-server 启动时**把账全打在日志里了**，一次加载就够：
    #   load_tensors:   CUDA0 model buffer size =  2375.91 MiB   权重
    #   llama_context:  n_ctx                 =      32768      本次请求的窗口
    #   llama_kv_cache: CUDA0 KV buffer size  =  2448.00 MiB   KV（对 ctx 线性）
    #   llama_kv_cache: size = 2448.00 MiB ( 32768 cells, 36 layers, ...)
    #   sched_reserve:  CUDA0 compute buffer size = 186.09 MiB 计算缓冲（近似固定）
    #   load_tensors:   offloaded 37/37 layers to GPU
    #
    # 于是 per_token_kv = kv_size / n_ctx，剩下的是纯算术。见 _estimate_ctx()。

    def _parse_vram_log(self) -> dict | None:
        """从**主权重本次加载**的日志区间解析显存账目。取不到关键项返回 None。

        区间锚定主模型（见 _main_load_section）—— 挂了草稿/塔时，"最后一条
        loading model tensors"属于它们，拿那段解析会把草稿的 buffer 当成
        主模型的（2026-10-16：估值因此彻底算不出来/算错）。

        KV 缓存可能是**两段**（gemma 系：非 SWA 段随 ctx 线性增长；SWA 段是
        滑动窗口、cells 固定不随 ctx 变）：
            llama_kv_cache: CUDA0 KV buffer size = 340.00 MiB  (32768 cells, 5 layers)
            llama_kv_cache: CUDA0 KV buffer size = 478.12 MiB  ( 4608 cells, 25 layers)
        只有 cells == n_ctx 的那段能按 token 外推（per_token_kv_mib），其余
        段合计按固定开销计入 kv_fixed_mib。没有任何段的 cells 等于 n_ctx 时
        （老版本只打短行）→ 退回旧的总量/n_ctx 算法，固定段记 0。

        同时解析**层归属与幽灵块**（供 _weights_fully_resident 判 residency）：
        offloaded N/M 原样记 on_gpu/total_layers（layer_gap = M-N）；
        `W model has unused tensor blk.N.xxx (size = ... bytes) -- ignoring`
        按**块号去重**计数（ignored_count，与 _gpu_offload_info 同语义），
        字节数精确求和（ignored_mib）——幽灵块永不参与计算，算热权重时
        要剔掉（WB seq 25：Ornith 683 MiB 里 145 MiB 是幽灵张量）。
        """
        sec = self._main_load_section()
        if not sec:
            return None
        start, end = sec
        chunk = list(self._logbuf)[start:end]

        out: dict = {}
        kv_parts: list[tuple[float, int]] = []      # (MiB, cells)
        ignored_blocks: set[str] = set()            # 被丢弃的块号（去重）
        ignored_bytes = 0                           # 幽灵张量字节总数
        print_info_seen = False                     # 主模型段有没有 print_info 块
        for ln in chunk:
            if "print_info:" in ln:
                print_info_seen = True
            # 带 cells 的详细行（b10951 实测格式）："llama_kv_cache: size = 340.00 MiB ( 32768 cells, 5 layers, ...)"
            # ⚠️ 不能加 "KV buffer" 前缀要求 —— 前缀在另一条短行上，加了就永远
            # 匹配不上（旧代码这条正则就是死代码，一直默默用最后一条短行）。
            m = re.search(r"size\s*=\s*([\d.]+)\s*MiB\s*\(\s*(\d+)\s*cells", ln)
            if m:
                kv_parts.append((float(m.group(1)), int(m.group(2))))
            m = re.search(r"n_ctx\s*=\s*(\d+)", ln)
            if m and "llama_context" in ln and "n_ctx" not in out:
                out["n_ctx"] = int(m.group(1))
            # MoE 身份（print_info 段，主模型）：n_expert=0/缺失 = 稠密模型。
            # 只认主模型段里的第一条 —— 草稿也会打自己的 n_expert（实测
            # gemma MTP 头是稠密的 n_expert=0），取错会把 MoE 判成稠密。
            m = re.search(r"n_expert\s*=\s*(\d+)", ln)
            if m and "n_expert" not in out:
                out["n_expert"] = int(m.group(1))
            m = re.search(r"n_expert_used\s*=\s*(\d+)", ln)
            if m and "n_expert_used" not in out:
                out["n_expert_used"] = int(m.group(1))
            m = re.search(r"CPU_Mapped model buffer size\s*=\s*([\d.]+)\s*MiB", ln)
            if m:
                out["cpu_mapped_mib"] = float(m.group(1))
            # 主模型段内第一条 model/compute buffer 就是主模型的
            #（草稿/塔的在段外；同段内也不会重复打）
            m = re.search(r"CUDA0 model buffer size\s*=\s*([\d.]+)\s*MiB", ln)
            if m and "model_mib" not in out:
                out["model_mib"] = float(m.group(1))
            m = re.search(r"CUDA0 compute buffer size\s*=\s*([\d.]+)\s*MiB", ln)
            if m and "compute_mib" not in out:
                out["compute_mib"] = float(m.group(1))
            # 层归属 + 幽灵块（W 行带精确字节数）
            m = re.search(r"offloaded (\d+)/(\d+) layers to GPU", ln)
            if m and "on_gpu" not in out:
                out["on_gpu"], out["total_layers"] = int(m.group(1)), int(m.group(2))
            # fit 对专家权重的切分份数（MoE 专家分配优化的辅助判据 + 日志用）：
            #   set ngl_per_device[0].(n_layer, n_part)=(31, 24)
            #   set ngl_per_device[0].(n_layer, n_part, overflow_type)=(31, 24, GATE)
            # **以最后一条为准** —— fit 会先试若干候选（前面几行是中间档）。
            # 刻意不匹配 ngl_per_device_high[0]：那是另一侧的备份方案，不是
            # 最终生效值（实测 scan_fitt_256.log：(31,24) 之后 _high 报 23，
            # 但真正生效的是 (31,24)）。缺这个字段只影响辅助判据 3 与日志。
            m = re.findall(
                r"ngl_per_device\[0\]\.\(n_layer,\s*n_part(?:,\s*\w+)?\)\s*=\s*"
                r"\((\d+),\s*(\d+)", ln)
            if m:
                out["n_part"] = int(m[-1][1])
            m = re.search(r"unused tensor (blk\.\d+)\.", ln)
            if m:
                ignored_blocks.add(m.group(1))
            m = re.search(r"unused tensor .*?\(size = (\d+) bytes\)", ln)
            if m:
                ignored_bytes += int(m.group(1))
        if "on_gpu" in out:
            out["layer_gap"] = out["total_layers"] - out["on_gpu"]
        out["ignored_count"] = len(ignored_blocks)
        out["ignored_mib"] = ignored_bytes / 1048576.0
        # print_info 块在不在，决定 _detect_moe 能不能采信"没有 n_expert"
        # 这个否定信号（块不在 = 引擎没开口，不是引擎说"稠密"）
        out["print_info_seen"] = print_info_seen
        # 视觉塔的显存：mtmd 自己管，日志里若有独立行就计入固定开销
        for ln in chunk:
            m = re.search(r"CUDA0 model buffer size\s*=\s*([\d.]+)\s*MiB", ln)
            if m and "mmproj" in ln.lower():
                out["mmproj_mib"] = float(m.group(1))

        if not kv_parts:
            # 老版本只打短行（不带 cells）→ 按总量均摊（旧行为）
            for ln in chunk:
                m = re.search(r"CUDA0 KV buffer size\s*=\s*([\d.]+)\s*MiB", ln)
                if m:
                    kv_parts.append((float(m.group(1)), 0))
        if not kv_parts or not out.get("n_ctx"):
            return None
        n_ctx = out["n_ctx"]
        out["kv_mib"] = sum(s for s, _ in kv_parts)
        scalable = [(s, c) for s, c in kv_parts if c == n_ctx]
        if scalable:
            s, c = scalable[-1]                    # 同段多条取最后（信息最全）
            out["per_token_kv_mib"] = s / c
            out["kv_fixed_mib"] = sum(s for s, c in kv_parts if c != n_ctx)
        else:
            out["per_token_kv_mib"] = out["kv_mib"] / n_ctx
            out["kv_fixed_mib"] = 0.0
        return out

    @staticmethod
    def total_vram_mib() -> int | None:
        """显卡总显存（MiB）。读不到返回 None（无 NVIDIA 卡 / 命令缺失）。"""
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=8,
                creationflags=subprocess.CREATE_NO_WINDOW)
            if r.returncode == 0 and r.stdout.strip():
                return int(r.stdout.strip().splitlines()[0])
        except Exception:
            pass
        return None

    def _vram_pct(self, acct: dict | None) -> float:
        """当前加载的显存占用百分比（权重 + KV + 计算缓冲 + 视觉塔 / 总显存）。

        fit 档位扫描的水位保护判据（方案 §5 判据 4）：越激进放进显存的专家
        越多，贴近显存顶部时 compute buffer 会被挤，**实测反而更慢** ——
        所以 > 88% 就停，别贴顶。账目键由 _parse_vram_log 给（别另造一套）。
        读不到总显存（无 NVIDIA 卡）→ 0，此时水位保护自然不触发（其余判据照常）。
        """
        total = self.total_vram_mib() or 0
        used = sum(float((acct or {}).get(k) or 0)
                   for k in ("model_mib", "kv_mib", "compute_mib", "mmproj_mib"))
        return used * 100.0 / total if total else 0.0

    def _estimate_ctx(self, cfg: dict, mem: dict | None = None,
                      moe: bool = False) -> dict | None:
        """由显存账目**算**出理论上能开的最大 ctx。

        算式：  可分配给 KV 的显存 = 总显存 - 权重 - 计算缓冲 - 安全余量
                最大 ctx = 该显存 / 每 token KV

        ⚠️⚠️ **这个算式一定会高估，别把结果直接用。**
        实测标定（8G 独显，llama.cpp b10951，KV=q8_0）：

            模型            未校准算式   实测临界    所需系数
            josie-4b         69,632      40,960      0.59
            qwen3.5-4b-ud   258,048     110,592      0.43

        高估的原因：llama.cpp 在显存占用接近满（实测 ~94%）时**不会报错**，
        而是把层悄悄赶到 CPU。日志里的 model/KV/compute 三个数也不含 CUDA
        context 本身（cuBLAS/cuDNN handle）与分配器的对齐 padding。

        所以这里乘一个 **校准系数** 把上界压到安全区。默认 0.42（本机两次实测
        里较保守的那个，保证不猜大）—— 宁可低估：低估只是少用一点窗口，
        猜大了会掉层到 CPU 直接变慢十倍。
        实测 0.42 的效果：qwen3.5-4b-ud 估 106496 → 全层在 GPU 命中，
        距真实临界 110592 只差 4%，**3 次加载收工**。
        显存特别紧张时可再降到 0.35；想开更大窗口可试 0.5~0.6。

        算式：  可分配给 KV 的显存 = 总显存 - 权重 - 计算缓冲 - 安全余量
                该显存先扣掉不随 ctx 变的 KV 段（gemma 系 SWA 段是滑动
                窗口、cells 固定），剩下按每 token KV 线性摊：
                最大 ctx = (可分配 - KV固定段) / 每 token KV

        前提是**权重全驻显存** —— 调用方（measure_ctx）已用
        _weights_fully_resident 把装不下的模型挡掉；账目里真带着"大半
        权重在内存"时这里也直接放弃估算（best=0 + reason）。
        mem 可由调用方传入（刚解析过的，避免重复扫日志）。

        moe=True 时**放开全驻前提**：MoE 混合卸载下 CPU_Mapped 的专家
        权重不占显存（llama.cpp 按需流入，见类内 MoE 段注释），显存里
        只有 model_mib 那部分，算式原样成立 —— 这正是 gemma-4-26B 能在
        8G 卡上算出 192K 窗口的原因。稠密模型仍走全驻前提。
        """
        mem = mem if mem is not None else self._parse_vram_log()
        if not mem:
            return None
        total = self.total_vram_mib()
        if not total:
            return None
        # 安全余量：驱动上下文 + 显存碎片 + 多 slot 预留
        reserve = float(cfg.get("ctx_vram_reserve_mib") or 512)
        # 校准系数：压掉算式的高估（见上面实测表）
        ratio = float(cfg.get("ctx_estimate_ratio") or 0.5)
        fixed = mem.get("model_mib", 0.0) + mem.get("compute_mib", 0.0) + mem.get("mmproj_mib", 0.0)
        for_kv = total - fixed - reserve
        if for_kv <= 0:
            return {"best": 0, "mem": mem, "total_vram": total, "reason": "权重已占满显存"}
        if not KaiwuManager._weights_fully_resident(mem) and not moe:
            return {"best": 0, "mem": mem, "total_vram": total,
                    "reason": "权重无法全驻显存（大半在内存 mmap），算式不适用"}
        per_token = float(mem.get("per_token_kv_mib") or 0.0)
        if per_token <= 0:
            return None
        kv_fixed = float(mem.get("kv_fixed_mib") or 0.0)
        raw = max(for_kv - kv_fixed, 0.0) / per_token
        # 未校准的物理上界（对齐 + 夹 n_ctx_train），供"估命中后再往上探一档"用
        ceiling = int(raw // 4096 * 4096)
        n_train = self._n_ctx_train()
        if n_train:
            ceiling = min(ceiling, n_train)
        ceiling = max(ceiling, 0)
        # 校准后的保守值：真正拿去验证的那个
        raw *= ratio
        best = int(raw // 4096 * 4096)
        if n_train:
            best = min(best, n_train)
        return {"best": max(best, 0), "mem": mem, "total_vram": total,
                "per_token_kv_mib": mem["per_token_kv_mib"], "for_kv_mib": for_kv,
                "reserve": reserve, "ratio": ratio, "n_ctx_train": n_train,
                "raw": raw, "ceiling": ceiling}

    def _n_ctx_train(self) -> int | None:
        """模型原生窗口 n_ctx_train（日志里读）。读不到返回 None。

        只认**主模型段**（见 _main_load_section）：挂了草稿时草稿也会打一条
        自己的 n_ctx_train（实测 MTP 头 131072 vs 主模型 262144），取错会把
        估值上界夹错。"""
        sec = self._main_load_section()
        lines = list(self._logbuf)
        rng = lines[sec[0]:sec[1]] if sec else lines
        pat = re.compile(r"n_ctx_train\s*=\s*(\d+)")
        for ln in reversed(rng):
            m = pat.search(ln)
            if m:
                return int(m.group(1))
        return None

    def measure_ctx(self, model: str, cfg: dict | None = None,
                    timeout: float | None = None) -> dict:
        """自动测量该模型「可持续」的最大上下文（每请求）。

        判据分模型形态（2026-10-17 修订，gemma-4-26B-A4B 真机复测）：
          · **稠密模型**：权重全驻显存（层数在 GPU 且不 mmap 到内存）的
            最大 ctx；未全驻的档位要**实测 tok/s 达稠密门槛**
            （ctx_dense_min_tps，默认 30）才算数 —— 每 token 要读全部
            权重，mmap 一半就慢一半，半速窗口不如换小的。
          · **MoE 模型**（日志 print_info n_expert > 0）：层数在 GPU 且
            **实测 tok/s 达门槛**（ctx_moe_min_tps，默认 15）的最大 ctx。
            权重在内存 mmap 对 MoE 不是病态：每 token 只读激活的几个专家，
            冷专家躺在页缓存里不拖速度。未全驻时**先扫 fit 余量**（-fitt，
            2026-10-20）：让 fit 自己按份切分专家权重，把原本空着的显存用起来，
            实测比把专家全赶内存（--cpu-moe，旧行为）快 9.7%~22%，窗口不减。
            扫描全失败才退回 --cpu-moe（ctx_moe_prefer_cpu）。全程由实测速度
            定夺，不猜。
            ⚠️ 两个门槛分开定（2026-10-17）：稠密小模型至少 30
            （ctx_dense_min_tps，含近空→满载折扣）；MoE 定 15 是
            为弱机器留路（gemma 级在 15~18 也能用），8.2 的悬崖档
            照样拦。混用一个数字必错一头。

        流程：
          0. 一次温和探测（ctx_probe_ctx，真实服务配置：视觉塔/MTP 草稿
             按配置挂载）—— 读日志显存账目 + 实测一次 tok/s
          0.5 MoE 且未全驻 → **扫描 fit 余量**（-fitt 4096→64 逐档实测，
             取最快档）；一档都没起来才退回 --cpu-moe 探测（2026-10-20）
          1. 权重就装不下（稠密）→ 给下限一次机会后诚实收尾（记录下限 +
             实测原因），一次加载都不多烧
          2. 算式 × 校准系数给出**保守估计**，加载验证这一档（MoE 带测速）
          3. 不命中 → 按 ctx_bisect_max_loads 上限做少量二分兜底

        以前是无条件二分，一个模型最多 10+ 次真实加载，慢到 ctx_auto 长期没人
        开。现在最常见路径 2~3 次（MoE 多一次 fit 扫描），且结果写进
        config.ctx_measured 后**同一量化档永不再测**（除非换显卡 / 改 KV 量化 /
        换 llama-server 版本，由 ctx_measured_env 指纹自动识别并重测；或切换
        到没测过的量化档）。

        本机实测（8G 独显，b10951，完整配置含草稿+视觉塔）：
          qwen3.5-4b-ud   3 次加载 20s → 106496（真值 110592，命中 96%）
          gemma-4-26B-A4B -fitt 256 @114688 → 31.38 tok/s（--cpu-moe @94208 只有 25.6）
          gemma-4-26B-A4B --cpu-moe @131072 → 16.3 tok/s（旧兜底形态）
          ⚠️ 早先记的 21.96 / 20.02 / 23.60 / 22.83 都是**冷态读数**（只丢 1 枪），
             2026-10-20 晚独立验收已作废，别再引用。

        结果写入 config.ctx_measured[模型@量化档] + ctx_measured_env[同键]
        （环境指纹）；MoE 选定 fit 档另写 config.fit_target[同键]=MiB，
        以后每次普通加载都带 `-fitt <MiB>`（与测量形态保持一致）；只有扫描
        全失败退回 --cpu-moe 时才改写 config.moe_offload[同键]="cpu"。
        ⚠️ 首次会真实启停推理服务若干次（每次一次加载；MoE 的 fit 扫描
        每档约 1.5 分钟）。
        """
        cfg = cfg or load_config()
        lo = int(cfg.get("ctx_auto_min") or 32768)
        hi = int(cfg.get("ctx_auto_max") or 131072)
        step = 4096
        per_try = float(timeout or min(float(cfg.get("startup_timeout_sec", 900) or 900), 180.0))
        # 首次探测用一个温和的窗口：够读出显存账目就行，不必一上来就顶格。
        # 顶格加载慢，而且过大的 KV 分配本身就可能挤掉层，把"本来能开"测成"开不了"。
        probe_ctx = int(cfg.get("ctx_probe_ctx") or 32768)
        # MoE 速度门槛：低于它的窗口再大也不叫"可持续"（2026-10-17 实证：
        # --cpu-moe@196608 挂草稿时 VRAM 贴顶，31/31 层都在 GPU 却只有
        # 1.18 tok/s —— 层数判据完全看不出来，只有真测能抓住）。
        min_tps = float(cfg.get("ctx_moe_min_tps") or 5.0)
        # 稠密模型的速度门槛（2026-10-17 用户拍板：小的稠密模型至少 30）。
        # 与 MoE 分开定：稠密全驻时 30+ 是常态、掉层就腰斩；MoE 每 token 只读
        # 激活专家，8G 卡上 20~26 就是甜点位，混用一个门槛必错一头。
        dense_min_tps = float(cfg.get("ctx_dense_min_tps") or 30.0)

        attempts: list[dict] = []
        best: int | None = None
        n_total: int | None = None
        last_acct: dict | None = None      # 最近一次加载的显存账目
        last_tps: float | None = None      # 最近一次加载的实测 tok/s
        last_npart: int | None = None      # 最近一次加载的 fit 专家切分份数
        best_acct: dict | None = None      # 入选那一档的账目（备注用）
        best_tps: float | None = None

        def attempt(ctx: int, moe_cpu: bool = False,
                    tps: bool = False, extra: list[str] | None = None,
                    stat: str = "max") -> tuple[bool, int | None, int | None]:
            """返回 (是否起来, 上 GPU 的层数, 总层数)；账目落在 last_acct。

            moe_cpu: 本次加载加 --cpu-moe（专家权重显式走内存）。
            tps: 就绪后真实测一次速（MoE 判据 / 诚实备注都要实测数）。
            extra: 追加在命令行末尾的额外旗标（fit 扫描传 ["-fitt","256"]）。
                   ⚠️ 必须一路透传 start → _start_inner → 命令行；漏一层就是
                   "扫描跑了但参数没生效"，日志里看不出来（方案 §6.1 坑 1）。
            stat: 测速统计量。"max" = 判这档能不能跑（默认）；"mean" = 比档位
                   优劣（fit 扫描用）—— 分工见 _measure_tps。
            """
            nonlocal last_acct, last_tps, last_npart
            self.stop(cfg)
            time.sleep(1.0)
            tag = (" --cpu-moe" if moe_cpu else ""
                   + (" " + " ".join(extra) if extra else ""))
            self._log(f"[测量] 试 {ctx}{tag}（单次上限 {per_try:.0f}s）…")
            ok = self.start(model, cfg, per_try, ctx_override=ctx, moe_cpu=moe_cpu,
                            extra_args=extra)
            info = self._gpu_offload_info() if ok else None
            on_gpu, total = info if info else (None, None)
            last_acct = self._parse_vram_log() if ok else None
            last_npart = int((last_acct or {}).get("n_part") or 0) or None
            last_tps = self._measure_tps(cfg, stat=stat) if (ok and tps) else None
            attempts.append({"requested": ctx, "ok": ok, "on_gpu": on_gpu,
                             "total": total,
                             "extra": list(extra) if extra else None,
                             "cpu_mapped": (last_acct or {}).get("cpu_mapped_mib"),
                             "n_part": last_npart,
                             "tps": last_tps})
            tail = ("" if not last_acct or "cpu_mapped_mib" not in last_acct
                    else f"，内存 mmap {last_acct['cpu_mapped_mib']:.0f} MiB")
            tps_s = f"，实测 {last_tps:.1f} tok/s" if last_tps else ""
            self._log(f"[测量]   {ctx}{tag} → " + ("起不来" if not ok else
                       f"GPU {on_gpu}/{total} 层{tail}{tps_s}"
                       + ("（全部在 GPU）" if on_gpu == total else "（有层掉到 CPU）")))
            return ok, on_gpu, total

        def fits(on_gpu, total) -> bool:
            """稠密判据：层数全在 GPU 且权重全驻显存，**或者**实测够快。

            后半句是 2026-10-17 补的，两段现实压出来的：
              · Ornith-1.5-9B Q4_K_M 在 8G 卡上 31/34 层、959 MiB / 5.2G
                = 18% 权重在内存，两个代理判据（层数齐、residency 齐）**都不
                满足**，可它**实测 31.3 tok/s** —— 比不少全驻模型还快。旧逻辑
                照着 10% 的比例阈值判它"病态"，扣个 ⚠ 还劝人换显卡。
              · 同一周 gemma 的教训：31/31 层全在 GPU 也能只有 1.18 tok/s。
            两条合起来说明白一件事：**层数和 residency 都是猜的代理指标，
            实测 tok/s 才是证据**。够快就放行。

            速度门槛用稠密专用的 ctx_dense_min_tps（默认 30，2026-10-17
            用户拍板"小的稠密模型至少 30"）。定 30 而不是 5，除了"半速
            不糊弄"之外还有一层：测速是**近空上下文**读数，填满后注意力
            扫更多 KV 还要再打折（本机实测降 43%）。30 的门槛等于要求
            满载也有 ~17 tok/s，折扣已经 baked in，不用另乘安全系数。

            保守边界（不能省）：tps 测不到时不放宽，仍按代理判据 —— 宁严勿宽，
            别把"没测成"当成"跑得动"。真正不行的档位会被二分里那些更大的
            候选真跑一次淘汰掉，用不着靠比例阈值提前替它做决定。
            """
            if not bool(total):
                return False
            if (on_gpu == total
                    and KaiwuManager._weights_fully_resident(last_acct)):
                return True
            return last_tps is not None and last_tps >= dense_min_tps

        def dense_partial(acct=None) -> bool:
            """这一档是"层数全在 GPU、但权重有一部分在内存"的稠密混合形态？

            用来决定写哪张卡：这种档位跑得挺快（证据说话），只能写 ℹ 信息，
            不能像真降级那样写 ⚠ 警告 + 换显卡。

            ⚠️ 必须传**入选那一档**的账目（best_acct）—— 二分里最后一次尝试
            常是"太大的失败档"，用 last_acct 会把掉层的账目当成结论。
            """
            acct = last_acct if acct is None else acct
            return (bool(acct)
                    and not KaiwuManager._weights_fully_resident(acct))

        def fits_moe(on_gpu, total) -> bool:
            """MoE 判据：层数全在 GPU + 实测速度达门槛。

            tps 测不到（请求失败/老版本没这字段）时只按层数判 —— 宁宽勿严，
            别把"没测成"当成"慢"。真正慢的档位会被下一个带 tps 的候选
            或二分抓住。
            """
            if not (bool(total) and on_gpu == total):
                return False
            if last_tps is None:
                return True
            return last_tps >= min_tps

        def scan_fit() -> dict | None:
            """MoE fit 余量定位：**先算后测** —— 预测几档、实测裁判，取最快的。

            返回选中档的记录 {fitt, tps, n_part, on_gpu, total, acct}；
            一档都没起来返回 None（调用方退回 --cpu-moe 兜底）。

            为什么必须测、不许写死一个值（方案 §5）：最优档随模型变 ——
            Gemma-4-26B 停在 256，LFM2.5-24B 要 128 才平，把 256 写死在
            LFM 上就少赚一截。

            ⚠️ 2026-10-21 改成「先算后测」（用户拍板，方案见 app/fit_scan.py、
            memorix #109/#110）：旧版是全量扫描 4096→64 七档，LFM2.5 实测
            **606 秒 = 10.1 分钟**，用户就在那儿干等；其中还有两个保守档
            （gemma 上是"层级别回填"）纯粹白测，占总时 39%。改法：
              ① **算**：从刚加载完的探测实例日志里解析上游 fit 的切分阶梯
                 （n_part → 占多少 MiB）+ 运行时空闲显存，**不额外花时间**。
              ② **定位置**：预算 = 空闲 − fitt，预测 n_part = 使 mem(n_part)
                 ≤ 预算的最小 n_part。n_part 越小 = 专家越多在显存 = 越快
                 ⇒ 预测 n_part 最小的那档就是预测最快的。两个结构迥异的 MoE
                 模型 12/12 零误差（gemma 31 层 / LFM2.5 41 层）。
              ③ **测左右**：只实测「预测最快」+ 左右各一邻档（最多 3 档）。
                 与已选档 n_part 相同的直接跳过（同一配置分不出速度差：实测
                 LFM 128/64 差 0.56 < 容差 1.2）。
              ④ **定案**：选档规则一个字不改（取最快；同分取保守档）。
            兜底（任何一条不满足就退回**全量扫描**，旧行为不会更差）：
              · 日志里没有阶梯（老版本二进制 / 解析失败）→ 全扫；
              · 计划里的档一个都没测起来（起不来/测不到）→ 全扫；
              · **预测与实测的 n_part 差超过 1 层** → 弃用先算后测，**从头**
                按旧顺序全扫（已测的档作废重测）。差 1 层以内**不算失准**
                ——阶梯是采样点，目标落在两采样点之间要线性插值，边界上差
                一层是外推误差（真机：单次加载的阶梯对最保守那档就差 1 层，
                用严格相等会让收益直接归零）；
              · **复核**：计划档测完后阶梯变完整了，回头重算被跳过那几档的
                预测 —— 预测变了就补测那档。因为当初「跳过」的理由是
                「跟已测档同配置」，而这个判断可能建立在不够准的阶梯上，
                错判的代价是白丢一层专家（实测每层 ≈249 MiB / 几个 tok/s），
                比多测一次贵得多。
            ⚠️ 数学只用来定位，**实测才是裁判** —— 预测永不参与选档。
            ⚠️ 阶梯依赖 ctx（LFM fitt512：@118784 → 26、@32768 → 21），所以只
              用**同窗口**加载出来的阶梯。计划在扫档前**一次性**算出（用探测
              那次加载的阶梯），实测**不改计划**（改计划会打乱「只测哪几档」
              的可解释性），但会用**累积后的阶梯**复核被跳过的档（见上面
              「复核」那条）。
            ⚠️ 端到端只测 2 档是**设计如此**，不是漏测 —— 剩下的档要么
              预测装不下（层级别回填，必然更慢），要么跟已测档同一套专家
              分配置（速度分不出来），要么预测贴顶（88% 水位）。每一档
              跳过都在日志里写了理由（`[测量] 跳过 -fitt N：…`）。

            ⚠️ 2026-10-20 晚按独立验收报告重写测量方法（旧版选低了档）：
              · **先补预热枪**（_FIT_WARM_SHOTS）：专家权重走 mmap 时页缓存
                要逐渐喂热，前 4~5 枪都还在爬坡。不预热的话"后跑的档天然更热"，
                读数被系统性地抬高。
              · **阈值 0.3 → _FIT_NOISE_TPS（1.2）**：0.3 是按单次采样噪声
                0.11 定的，而跨加载噪声实测 **1.19**（同配置两次加载差 1.19），
                0.3 分辨不了任何档位。
              · **停止判据补上 `n_part` 不再下降**（方案 §5 判据 3 原先没实现）：
                只有"更激进却没多放进任何专家 + 也没快过噪声"才算到顶。
                只按速度停会过早收工 —— 报告自己的稳态曲线里 1024→512 只
                +0.55，按 1.5 的阈值会直接停、永远到不了最优的 256。

            全扫时的停止判据（顺序即优先级，仅在兜底路径生效）：
              ① 本档起不来 → 停（更激进只会更差）
              ② 本档测不到速度 → 停（无法比档位优劣，宁可不选也别瞎选）
              ③ 显存 > 88% → 停（水位保护：贴顶会让 compute buffer 被挤，
                 实测反而更慢）
              ④ `n_part` 没有比上档再降（没多放进专家）**且** 增益 < 噪声
                 → 停（到物理上限了）
            六/七档是兜底上限；本机 gemma 实测 5~6 档收工。

            选中规则：先取实测最快；**与最快相差在噪声以内的档，取最保守的
            （fitt 数值最大）** —— 差值本身分不出来，就不该为了噪声把更多
            专家塞进显存（那会挤掉大窗口的余量）。本机实测 128(35.95) 与
            256(35.46) 差 0.49 < 1.2 → 记 256。
            """
            # 速度差容差：配置值（界面「速度差容差」），缺省用本机实测 1.2
            gain_min = self._fit_gain_min(cfg)
            # 预热：现成的探测负载上补两枪，把 11.3G 权重喂进页缓存
            self._log(f"[测量] 扫档前预热 {_FIT_WARM_SHOTS} 枪"
                      "（专家权重走内存 mmap，前几枪都在爬坡）…")
            self._measure_tps(cfg, rounds=_FIT_WARM_SHOTS, drop=0)
            self._log(f"[测量] 速度差容差 {gain_min:.2f} tok/s"
                      "（两档差小于它视为一样快）")
            results: list[dict] = []
            prev_npart: int | None = None

            def one(fitt: int) -> str:
                """实测一档并记账。返回 "ok" / "fail"（起不来或测不到）/ "hot"（超水位）。

                账目写在 last_acct/last_tps/last_npart（attempt 的副作用），
                记录追加进 results —— 与全扫时的行为逐字一致，只是把"停"
                交给调用方决定（先算后测路径要跳过失败档继续测邻档）。
                """
                ok, on, tot = attempt(probe_ctx, tps=True,
                                      extra=["-fitt", str(fitt)], stat="mean")
                if not ok:
                    self._log(f"[测量] 扫档 -fitt {fitt} 起不来"
                              "（更激进只会更差）")
                    return "fail"
                if last_tps is None:
                    self._log(f"[测量] 扫档 -fitt {fitt} 测不到速度"
                              "（无法比档位优劣）")
                    return "fail"
                results.append({"fitt": fitt, "tps": last_tps,
                                "n_part": last_npart, "on_gpu": on, "total": tot,
                                "acct": last_acct})
                exp_s = (f"，n_part={last_npart}"
                         f"（{max(int(tot or 0) - int(last_npart), 0)} 层在显存）"
                         if last_npart else "")
                self._log(f"[测量] 扫档 -fitt {fitt} → {last_tps:.2f} tok/s"
                          f"{exp_s}，显存 {self._vram_pct(last_acct):.0f}%")
                if self._vram_pct(last_acct) > 88:
                    self._log("[测量] 扫档：显存 > 88%（水位保护）")
                    return "hot"
                return "ok"

            # ============ 先算：从「刚加载完的探测实例」日志里取切分阶梯 ============
            # 免费（日志已经在 self._logbuf 里），不额外加载。阶梯是上游 fit
            # 自己试过的每个候选（n_part → 占多少 MiB），见 app/fit_scan.py。
            pred = fit_scan.parse_fit_ladder(self.log_text())
            ladder: dict = dict(pred["ladder"] or {})
            free_mib = pred["free_mib"]
            n_layer = int((last_acct or {}).get("total_layers")
                          or pred.get("n_layer") or 0)
            total_mib = self.total_vram_mib()
            plan = fit_scan.plan_scan(ladder, free_mib, _FIT_CANDIDATES,
                                      n_layer=n_layer, total_vram_mib=total_mib)
            measured: set[int] = set()      # 已经真测过的档（补扫时用）
            full_sweep = not plan["plan"]
            if plan["plan"]:
                self._log("[测量] 先算后测：切分阶梯 "
                          + "，".join(f"{n}:{m}MiB" for n, m in sorted(ladder.items()))
                          + f"，运行时空闲显存 {free_mib} MiB")
                self._log("[测量] 预测 " + "；".join(
                    f"-fitt {p['fitt']} → n_part {p['n_part']}"
                    f"（{p['mem']:.0f} MiB/{p['pct']:.0f}%）" if p["pct"] is not None
                    else f"-fitt {p['fitt']} → n_part {p['n_part']}"
                    for p in plan["plan"])
                    + f" —— 只测这 {len(plan['plan'])} 档")
                for f, why in plan["skipped"].items():
                    self._log(f"[测量] 跳过 -fitt {f}：{why}")
                for p in plan["plan"]:
                    st = one(p["fitt"])
                    measured.add(int(p["fitt"]))
                    if st != "fail" and last_npart is not None:
                        # 容忍 ±1 层：阶梯是采样点，目标落在两采样点之间时要
                        # 线性插值，边界上差一层是外推误差，不是模型/日志与
                        # 公式不符（真机：单次加载的阶梯对最保守那档就差 1 层，
                        # 用严格相等会让收益直接归零）。
                        if abs(int(last_npart) - int(p["n_part"])) > 1:
                            # 预测不是裁判：差超过一层 → 后面的"跳过"决定不可信。
                            # **把已测的档全部作废、从头按旧顺序重扫** ——
                            # 否则会变成「先激进后保守」的混合顺序，页缓存热度
                            # 偏差方向翻转，选出的档可能与旧路径不同。
                            self._log(
                                f"[测量] 预测失准：-fitt {p['fitt']} 预测 "
                                f"n_part={p['n_part']}、实测 {last_npart}"
                                "（差 >1 层）→ 弃用先算后测，从头全量扫描"
                                "（预测只用来定位，不参与选档）")
                            results.clear()
                            measured.clear()
                            prev_npart = None
                            full_sweep = True
                            break
                    # 每加载一档就把它的阶梯并进来（阶梯随目标变大而变丰富）
                    more = fit_scan.parse_fit_ladder(self.log_text())
                    ladder = fit_scan.merge_ladder(ladder, more["ladder"] or {})
                    if st in ("fail", "hot"):
                        break

                # ---- 复核被跳过的档：当初的「跳过」可能是基于浅阶梯做的 ----
                # 阶梯随目标变大而变丰富：计划档测完之后，我们手上的阶梯比
                # 算计划时更全。**某档的预测 n_part 变了** → 当初「它跟已测
                # 档同配置、分不出速度差」的判断可能是错的 → 补测那档。
                # 不补测的代价是可能白丢一层专家（实测每层专家 ≈249 MiB，
                # 对应几个 tok/s），比多测一次贵得多 —— 宁可多测。
                if not full_sweep and results:
                    recheck: list[int] = []
                    for fitt, why in plan["skipped"].items():
                        if "层级别回填" in why:
                            continue          # 这类跟阶梯精度无关，永远不合法
                        p2, _ = fit_scan.predict_n_part(
                            ladder, free_mib, fitt, n_layer)
                        old = (plan.get("preds") or {}).get(fitt)
                        if p2 is not None and old is not None and p2 != old:
                            recheck.append(int(fitt))
                    for fitt in sorted(recheck, reverse=True):
                        if fitt in measured:
                            continue
                        self._log(f"[测量] 补测：-fitt {fitt} 的预测 n_part "
                                  f"随阶梯变完整而改变（{(plan.get('preds') or {}).get(fitt)}"
                                  f" → 新预测）→ 当初的跳过可能错判，补测这一档")
                        st = one(fitt)
                        measured.add(fitt)
                        if st in ("fail", "hot"):
                            break
                if not results:
                    self._log("[测量] 先算后测计划的档一个都没测起来"
                              "（起不来或测不到速度）→ 回退全量扫描")
                    full_sweep = True
            else:
                self._log(f"[测量] 先算后测不可用（{plan['reason']}）"
                          " → 全量扫描 4096→64")
            if full_sweep:
                for fitt in _FIT_CANDIDATES:
                    if fitt in measured:
                        continue
                    st = one(fitt)
                    measured.add(int(fitt))
                    if st != "ok":
                        break
                    # 到顶判据：没多放进专家，且速度也没快过噪声
                    if len(results) >= 2:
                        gain = results[-1]["tps"] - results[-2]["tps"]
                        if (prev_npart is not None and last_npart is not None
                                and last_npart >= prev_npart
                                and gain < gain_min):
                            self._log(f"[测量] 扫档：n_part 不再下降"
                                      f"（{prev_npart}→{last_npart}）且增益 "
                                      f"{gain:+.2f} < 容差 {gain_min:.2f} → 停")
                            break
                    prev_npart = last_npart
            if not results:
                return None
            top = max(results, key=lambda r: r["tps"])
            # 同分（差在噪声内）取最保守的档：不拿噪声去换显存
            tied = [r for r in results
                    if r["tps"] >= top["tps"] - gain_min]
            best_r = max(tied, key=lambda r: r["fitt"])
            others = [r for r in results if r is not best_r]
            if others:
                alt = max(others, key=lambda r: r["tps"])
                self._log(f"[测量] fit 档位选定 {best_r['fitt']}"
                          f"（实测 {best_r['tps']:.2f}"
                          + (f"；最快是 {top['fitt']} = {top['tps']:.2f}，"
                             f"差 {top['tps'] - best_r['tps']:.2f} 在容差"
                             f" {gain_min:.2f} 内 → 取更保守的"
                             if top is not best_r else
                             f"，次优 {alt['fitt']} = {alt['tps']:.2f}")
                          + "）")
            else:
                self._log(f"[测量] fit 档位选定 {best_r['fitt']}"
                          f"（实测 {best_r['tps']:.2f}）")
            return best_r

        # ============ 第 0 步：一次温和加载，读出显存账目 + 实测速度 ============
        ok0, on0, tot0 = attempt(probe_ctx, tps=True)
        if not ok0:
            self._error = "自动测量失败：连探测窗口都起不来（模型可能损坏或显存严重不足）"
            self._log("[测量] " + self._error)
            return {"model": model, "best": None, "attempts": attempts}
        n_total = tot0

        # ============ 第 0.5 步：MoE → 混合卸载探测 + 判据切换 ============
        # 2026-10-17：gemma-4-26B-A4B 在 8G 卡上从未"全驻"过（15.1G 权重 vs
        # 8G 显存），旧逻辑就此判死刑。真相是专家走内存 mmap 后 18.7 tok/s、
        # 窗口还能开到 131072+ —— 混合卸载才是小显存大 MoE 的正常形态。
        # ⚠️ 2026-10-17 午后修的隐蔽 bug：混合探测的条件原来写成
        # "moe and not fits(on0, tot0)"，而 fits 的稠密分支含"实测 tok/s 达
        # 稠密门槛"—— 默认卸载恰好够快（gemma 默认卸载实测 31.8 tok/s，
        # 高于稠密门槛 30）时就跳过混合探测，判据还落回稠密：30 门槛下
        # gemma 被压到 32768，备注错写成"稠密混合"（12:30 那次 196608
        # 也是走稠密门槛 5 蒙对的，同样错）。MoE 的判据只该由身份决定；
        # 混合探测的条件是"未全驻"，与默认卸载快不快无关。
        # 模型形态判定（三级判据，见 _detect_moe）：引擎自报 n_expert 为权威，
        # 名称标记只是引擎没开口时的兜底。日志里打出来，用户能亲眼看见每个
        # 模型进了哪个桶 —— "永远判断正确"要靠可见，不靠相信。
        moe = KaiwuManager._detect_moe(last_acct, model)
        n_exp = int((last_acct or {}).get("n_expert") or 0)
        n_exp_used = int((last_acct or {}).get("n_expert_used") or 0)
        if n_exp > 0:
            self._log(f"[测量] 模型形态：MoE（引擎自报 n_expert={n_exp}"
                      + (f"，每 token 激活 {n_exp_used}" if n_exp_used else "")
                      + "）→ 判据用 MoE 门槛")
        elif (last_acct or {}).get("print_info_seen"):
            self._log("[测量] 模型形态：稠密（引擎自报无 n_expert）"
                      "→ 判据用稠密门槛")
        else:
            self._log("[测量] 模型形态：引擎未自报架构，名称无 MoE 标记，"
                      "按稠密处理（保守方向）→ 判据用稠密门槛")
        moe_cpu = False
        fit_target: int | None = None      # 扫描选定的 fit 余量（退回 --cpu-moe 时为 None）
        if moe:
            # ---- MoE 专家分配：扫描 fit 余量（2026-10-20 方案，替换旧 --cpu-moe）----
            # 旧逻辑一见"权重装不进显存"就把**全部**专家赶进内存（--cpu-moe），
            # 显存空着 3G 没人用，白亏 9.7%（gemma）/ 22.1%（LFM）。真相：
            # 不传任何 MoE 参数时 fit 的 n_part 会**每层内部按份均分**专家权重，
            # 那才是最快的切法（Ollama 用的就是它）。`-fitt` 只控制给系统留多少
            # 余量：**数值越小越激进，放进显存的专家越多**。
            # ⚠️ 最优档随模型变（Gemma 停在 256、LFM 要 128 才平），**不许写死** ——
            # 所以这里扫档实测择优（方案 §5）。扫不动/老二进制没 -fitt → 退回
            # --cpu-moe（旧路径原样保留，是回滚路径）。
            if (not KaiwuManager._weights_fully_resident(last_acct)
                    and self._llama_supports(cfg, "-fitt")):
                self._log("[测量] MoE 模型且权重未全驻 → 扫描 fit 余量"
                          "（-fitt：4096 保守 → 64 激进，实测择优）…")
                picked = scan_fit()
                if picked:
                    fit_target = int(picked["fitt"])
                    # 基线换成"选中那一档"的账目/层数/速度 —— 扫描最后一档
                    # 未必是最快的档（方案：取最快，不是最后一档），拿它当
                    # 结论就会把档位选错、备注也写错。
                    last_acct = picked["acct"]
                    last_tps = picked["tps"]
                    last_npart = picked["n_part"]
                    on0, tot0 = picked["on_gpu"], picked["total"]
            if (fit_target is None
                    and cfg.get("ctx_moe_prefer_cpu", True)
                    and self._llama_supports(cfg, "--cpu-moe")):
                # fit 一档都没起来（或老二进制不认 -fitt）→ 退回旧兜底 --cpu-moe
                self._log("[测量] fit 余量扫描没能选定 → 退回 --cpu-moe 混合卸载"
                          "（专家权重走内存，显存全留给 KV/计算）…")
                acct_keep = last_acct
                ok_c, on_c, tot_c = attempt(probe_ctx, moe_cpu=True, tps=True)
                if ok_c:
                    moe_cpu = True
                    on0, tot0 = on_c, tot_c
                else:
                    last_acct = acct_keep   # 混合档没起来 → 回到默认卸载账目
            # 判据切换：MoE 不再要求全驻（无论 --cpu-moe 成没成）
            gate = fits_moe
            gate_kind = "moe"
        else:
            gate = fits
            gate_kind = "dense"
        # 扫描选中的档位必须**跟着后续每一次加载**（方案 §改动 3）：二分找窗口
        # 时若换回默认卸载/--cpu-moe，测出的窗口在"fit 切分"形态下不成立 ——
        # 那测的就不是同一个配置了。没扫描（稠密 / 退回 --cpu-moe）时为 None。
        scan_extra = ["-fitt", str(fit_target)] if fit_target else None

        # ============ 第 1 步：基线档不可持续 → 下限一次机会 + 诚实收尾 ============
        # 稠密模型：ctx 越大 KV 越大、留给权重的显存越少 —— 探测档都装不下时
        # 往上顶没有意义（2026-10-16：gemma 旧逻辑在这里被 "offloaded 31/31"
        # 骗着烧 3 次加载，最后写下 102400 的错值）。MoE：混合卸载也摸不到
        # 速度门槛时同理，但备注写实测数据，不写编造的"2~3 tok/s"。
        if not gate(on0, tot0):
            total_vram = self.total_vram_mib() or 0
            # 探测档比下限大时，再给下限一次机会（KV 更小，也许能全驻/够快）
            if probe_ctx > lo:
                ok_l, on_l, tot_l = attempt(lo, moe_cpu=moe_cpu, tps=True,
                                            extra=scan_extra)
                if ok_l and gate(on_l, tot_l):
                    best, best_acct, best_tps = lo, last_acct, last_tps
                    if gate_kind == "moe":
                        note = KaiwuManager._note_moe_hybrid(
                            best_acct, best_tps, lo, fit=fit_target)
                    else:
                        # 稠密却有一部分权重在内存（靠实测放行的）：写 ℹ 信息，
                        # 不能当降级处理 —— 比例是猜的，速度是跑出来的
                        note = (KaiwuManager._note_dense_partial(
                            best_acct or {}, best_tps, lo, dense_min_tps)
                            if dense_partial(best_acct) else "")
                    self._log(f"[测量] ✓ {lo} 可用"
                              f"（{f'{best_tps:.1f} tok/s' if best_tps else '全层在 GPU'}）")
                    self._save_measured(cfg, model, best, note=note,
                                        mode="cpu" if moe_cpu else "",
                                        fit=fit_target)
                    return {"model": model, "best": best, "attempts": attempts,
                            "loads": len(attempts), "probe": probe_ctx}
            # 诚实降级：备注全部来自实测
            if gate_kind == "moe":
                reason = KaiwuManager._note_moe_degraded(
                    last_acct or {}, last_tps, lo, min_tps)
            else:
                reason = KaiwuManager._note_dense_degraded(
                    last_acct or {}, last_tps, lo, total_vram, dense_min_tps)
            self._log(f"[测量] ✗ {reason}")
            self._log(f"[测量] {model} 连 {lo} 都无法可持续，按 {lo} 记录"
                      f"（GPU {on0}/{tot0} 层）")
            self._save_measured(cfg, model, lo, note=reason,
                                mode="cpu" if moe_cpu else "",
                                fit=fit_target)
            return {"model": model, "best": lo, "attempts": attempts,
                    "loads": len(attempts), "degraded": True,
                    "reason": reason, "probe": probe_ctx}

        # 探测窗口通过判据 → 以它为基线往上试
        best = probe_ctx
        best_acct, best_tps = last_acct, last_tps

        # 估值的 residency 前提对"混合形态"一律不适用 —— 不只 MoE。稠密模型若
        # 靠实测 tok/s 放行（权重有一部分在内存），账目里的 model_mib 同样是
        # **驻在显存**的那部分，CPU_Mapped 不占显存，算式原样成立。
        # （2026-10-17：Ornith 31/34 层 + 959 MiB 走内存，估值得照算，否则这类
        # 模型永远只能拿 ctx_auto_min 的下限值。）
        bypass_residency = bool(moe) or dense_partial(last_acct)
        est = self._estimate_ctx(cfg, mem=last_acct, moe=bypass_residency)
        est_val = 0
        if est:
            est_val = int(est.get("best") or 0)
            if est.get("reason"):
                self._log(f"[测量] {est['reason']}")
            # reason 分支（显存已满 / 算式不适用）只带 mem，没有下面这些
            # 推算字段 —— 以前稠密模型根本走不到这里（未全驻就在第 1 步被
            # 降级掉了），2026-10-17 起稠密也能靠实测放行，这个死角才暴露。
            m = est["mem"]
            per_tok = float(est.get("per_token_kv_mib") or 0.0)
            if per_tok <= 0:
                self._log(f"[测量] 显存账目：权重 {m.get('model_mib', 0):.0f} MiB"
                          + (f"（内存 mmap {m.get('cpu_mapped_mib', 0):.0f} MiB）"
                             if m.get("cpu_mapped_mib") else "")
                          + f" + KV {m.get('kv_mib', 0):.0f} MiB@{m.get('n_ctx')}"
                          f"（固定段 {m.get('kv_fixed_mib', 0):.0f} MiB）"
                          f" + 计算缓冲 {m.get('compute_mib', 0):.0f} MiB，"
                          f"总显存 {est['total_vram']} MiB")
                self._log("[测量] 估不出每 token KV（账目里没有可外推的线性 KV 段），"
                          "退回顶格试一次")
                est_val = 0
            else:
                self._log(
                    f"[测量] 显存账目：权重 {m.get('model_mib', 0):.0f} MiB"
                    + (f"（内存 mmap {m.get('cpu_mapped_mib', 0):.0f} MiB）"
                       if m.get("cpu_mapped_mib") else "")
                    + f" + KV {m.get('kv_mib', 0):.0f} MiB@{m.get('n_ctx')}"
                    f"（固定段 {m.get('kv_fixed_mib', 0):.0f} MiB）"
                    f" + 计算缓冲 {m.get('compute_mib', 0):.0f} MiB，"
                    f"总显存 {est['total_vram']} MiB")
                self._log(
                    f"[测量] 每 token KV = {per_tok * 1024:.1f} KiB，"
                    f"预留 {est.get('reserve', 0):.0f} MiB × 校准 "
                    f"{est.get('ratio', 0):.2f} → 估 {est_val}")

        # ============ 第 2 步：算式给上界 → 一次加载验证 ============
        # 算式是**上界**（有算不见的开销），所以只把它当"值得一试的候选"，
        # 命中与否由那次加载说了算。命中就 2 次加载收工（探测 + 验证）。
        # 二分预算只数"基线定下来之后"的加载 —— 探测（MoE 还有一次
        # --cpu-moe 探测）是读账目的固定成本，不该吃掉用户的二分预算。
        search_loads = 0
        if est_val > 0 and est_val > lo and (best is None or est_val > best):
            guess = est_val
            self._log(f"[测量] 用估值 {guess} 验证一次…")
            search_loads += 1
            ok, on_gpu, total = attempt(guess, moe_cpu=moe_cpu, tps=True,
                                        extra=scan_extra)
            if total:
                n_total = total
            if ok and gate(on_gpu, total):
                best, best_acct, best_tps = guess, last_acct, last_tps
                lay_s = ("全层在 GPU" if total and on_gpu == total
                         else f"{on_gpu}/{total} 层（其余在 CPU，靠实测放行）")
                self._log(f"[测量] ✓ {guess} 可用（{lay_s}"
                          f"{f'，{best_tps:.1f} tok/s' if best_tps else ''}）")
                # 估值命中后给二分抬上界：max(ctx_auto_max, 估值+25%)，再夹
                # 未校准物理上界（ceiling，已夹 n_ctx_train）。两个作用：
                # ① 估值超过 ctx_auto_max 时（大 MoE 常态），hi 若留在
                #    ctx_auto_max，`best < hi` 为假，"命中后再顶一两档"的
                #    预算一分都花不出去；
                # ② 校准系数 + probe 账目的保守性实打实存在 —— 2026-10-17
                #    实测 gemma：估值 163840，手动顶到 192512 仍 20.2 tok/s
                #    （VRAM 6576/8151）。顶档就是把这 15% 找回来。
                # 只抬 25% 而不是直接到 ceiling：ceiling 可能是估值的两倍
                # （稠密模型校准压得狠），全给它会让二分预算全烧在绝对到不了
                # 的高档上（实测过：3 次预算耗光、best 原地不动）。
                ceil_v = int((est or {}).get("ceiling") or 0)
                hi = max(hi, min(ceil_v, guess * 5 // 4))
            else:
                # 校准后仍高估（显存特别紧张 / MoE 速度不达标）→ 以这次为新上界
                self._log(f"[测量] 估 {guess} 不可用 → 二分收敛")
                hi = guess - step
        elif not est_val:
            # 算式算不出来（读不到显存账目 / 非 NVIDIA 卡 / nvidia-smi 缺失）
            # → 退回"顶格试一次"，至少比默认值 32768 好
            self._log(f"[测量] 读不到显存账目，退回顶格 {hi} 试一次…")
            search_loads += 1
            ok, on_gpu, total = attempt(hi, moe_cpu=moe_cpu, tps=True,
                                        extra=scan_extra)
            if total:
                n_total = total
            if ok and gate(on_gpu, total):
                best, best_acct, best_tps = hi, last_acct, last_tps
                self._log(f"[测量] ✓ {hi} 可用")
            else:
                hi = hi // 2

        # ============ 第 3 步：二分收敛（默认不做）==========
        # 除非显式要求追最大值，否则**算式命中就到此为止**。
        # 理由：算式已经落在安全区（校准 0.5 留足垫子），再往上顶那几千 token
        # 对实际体验几乎无感，却要每档多等一次加载（本机 4B 每次 ~7s，追满要
        # 7 次 ≈ 50s，而结果只比估值多 3%）。首次测量这笔账要算"够用就好"。
        max_loads = int(cfg.get("ctx_bisect_max_loads") or 0)
        if max_loads > 0 and (best is None or best < hi):
            lo_b, hi_b = max(lo, best or lo), hi
            while lo_b <= hi_b and search_loads < max_loads:
                mid = max(lo_b, (lo_b + hi_b) // 2 // step * step)
                if best is not None and mid <= best:
                    lo_b = mid + step
                    continue
                search_loads += 1
                ok, on_gpu, total = attempt(mid, moe_cpu=moe_cpu, tps=True,
                                            extra=scan_extra)
                if total:
                    n_total = total
                if ok and gate(on_gpu, total):
                    best, best_acct, best_tps = mid, last_acct, last_tps
                    lo_b = mid + step            # 还能再大点
                else:
                    hi_b = mid - step            # 太大（掉层/起不来/太慢）

        if best:
            # ---- 方案 §6.2：在**最终窗口**下复核 fit 档位 ----
            # 扫描发生在 probe_ctx（KV 小、可留给专家的显存更多），而最终窗口
            # 通常更大（KV 更大、余量更紧）：同一个档位换了窗口就不再是同一个
            # 配置，probe 档下最快的余量到了最终窗口未必还是最快。所以窗口明显
            # 大于探测窗口时，用最终窗口再比一次**相邻（更激进）**的那一档。
            # 只在两档都过判据、且更激进档真的更快（超过「速度差容差」
            # fit_gain_min_tps，本机默认 1.2）时才替换 —— 0.3 那种阈值分不出档位。
            # 代价 +1~2 次加载；只在"测量"时发生一次。
            if (fit_target and best > probe_ctx and best_tps
                    and fit_target in _FIT_CANDIDATES):
                i = _FIT_CANDIDATES.index(fit_target)
                rival = _FIT_CANDIDATES[i + 1] if i + 1 < len(_FIT_CANDIDATES) else None
                if rival is not None:
                    cur_tps = float(best_tps)
                    self._log(f"[测量] 最终窗口 {best} 下复核 fit 档位："
                              f"{fit_target} vs 更激进 {rival}…")
                    ok_r, on_r, tot_r = attempt(best, tps=True,
                                                extra=["-fitt", str(rival)],
                                                stat="mean")
                    if ok_r and gate(on_r, tot_r) and last_tps \
                            and last_tps > cur_tps + self._fit_gain_min(cfg):
                        fit_target = rival
                        best_acct, best_tps = last_acct, last_tps
                        self._log(f"[测量] 最终窗口复核：改用更激进档 {rival}"
                                  f"（{last_tps:.2f} > {cur_tps:.2f} tok/s）")
                    else:
                        why = ("起不来" if not ok_r else
                               ("过不了判据" if not gate(on_r, tot_r) else "没更快"))
                        tps_s = f"{last_tps:.2f}" if last_tps else "未测到"
                        self._log(f"[测量] 最终窗口复核：{rival} {why}"
                                  f"（{tps_s} vs {cur_tps:.2f}）→ 保持 {fit_target}")
                        # 复核那次把服务换成了 rival 的形态 → **必须**再用选中档
                        # 把服务与账目恢复回来：记录的形态必须与实测/服役形态一致，
                        # 否则"二分测的窗口"与"以后加载的形态"就是两套配置。
                        ok_c, on_c, tot_c = attempt(
                            best, tps=True,
                            extra=["-fitt", str(fit_target)], stat="mean")
                        if ok_c and gate(on_c, tot_c):
                            best_acct, best_tps = last_acct, last_tps
                            n_total = tot_c or n_total
                        else:
                            # 理论上到不了（这档刚才是入选档）——真出现就明说，
                            # 别把一次没复核成功的结果当结论写进备注
                            self._log("[测量] ⚠ 复核后回不到所选档，备注按"
                                      "较早那次实测写")
            note = ""
            mode = "cpu" if moe_cpu else ""
            if gate_kind == "moe" and best_acct:
                # MoE 混合形态值得写明白：用户看到 10G 权重跑在 8G 卡上时
                # 有权知道"专家怎么摆的 + 实测速度"，而不是一句警告。
                note = KaiwuManager._note_moe_hybrid(best_acct, best_tps, best,
                                                     fit=fit_target)
            elif gate_kind == "dense" and dense_partial(best_acct):
                # 稠密混合形态同理：一部分权重走内存但实测够快 → ℹ 信息
                note = KaiwuManager._note_dense_partial(
                    best_acct or {}, best_tps, best, dense_min_tps)
            # 层数按**入选那一档**的账目报，不要想当然写成 N/N（2026-10-17：
            # Ornith 是 31/34 靠实测放行的，写"34/34 层"就是在吹牛）
            lo_l = int((best_acct or {}).get("on_gpu") or 0)
            hi_l = int((best_acct or {}).get("total_layers") or 0)
            lay_summary = (f"{lo_l}/{hi_l} 层在 GPU" if hi_l and lo_l != hi_l
                           else f"GPU {n_total}/{n_total} 层")
            self._log(f"[测量] {KaiwuManager._ctx_cache_key(cfg, model)} "
                      f"可持续最大上下文 = {best}"
                      f"（{lay_summary}"
                      f"{f'，{best_tps:.1f} tok/s' if best_tps else ''}"
                      f"{f'，fit {fit_target}' if fit_target else ''}，"
                      f"共 {len(attempts)} 次加载）")
            self._save_measured(cfg, model, best, note=note, mode=mode,
                                fit=fit_target)
            # 回读校验：磁盘记录 vs 服务实际在跑的 ctx，不一致当场警告
            try:
                served = self.granted_ctx(cfg)
                if served and int(served) != int(best):
                    self._log(f"[测量] ⚠ 回读校验：记录 {best} 但服务实际跑在 {served}")
            except Exception:
                pass
            return {"model": model, "best": best, "attempts": attempts,
                    "loads": len(attempts), "estimate": est_val or None,
                    "probe": probe_ctx}
        # 全部候选都掉层/起不来/太慢 → 退回下限并如实记录
        ok, on_gpu, total = attempt(lo, moe_cpu=moe_cpu, tps=True,
                                    extra=scan_extra)
        if ok:
            note = f"仅 {lo} 及更小窗口能保持全层在显存，更大窗口会掉层/装不下"
            self._log(f"[测量] {model} 连 {lo} 都装不下全层，按 {lo} 记录"
                      f"（GPU {on_gpu}/{total} 层）")
            self._save_measured(cfg, model, lo, note=note,
                                mode="cpu" if moe_cpu else "",
                                fit=fit_target)
            return {"model": model, "best": lo, "attempts": attempts,
                    "loads": len(attempts), "degraded": True, "note": note}
        self._error = "自动测量失败：所有候选值都起不来"
        self._log("[测量] " + self._error)
        return {"model": model, "best": None, "attempts": attempts, "loads": len(attempts)}

    def _save_measured(self, cfg: dict, model: str, ctx: int, note: str = "",
                       mode: str = "", fit: int | None = None) -> None:
        """把测量结果写回 data/config.json（以磁盘现状为基准，避免覆盖别处改动）。

        键 = 「模型@量化档」（_ctx_cache_key）：不同量化体积不同、能开的
        窗口不同，一个模型每个档各记一条，互不沿用。同时记一份同键的
        ctx_measured_env（测量时的显卡名+总显存+kv 量化档+llama 版本+
        伴生挂载状态）—— 以后加载同一量化档时先比对这个"环境指纹"，
        换了显卡 / 改了 KV 量化 / 换了 llama-server 版本 / 开关了视觉塔
        或 MTP 草稿，旧缓存就不再可信，自动重测。
        用户换机器时这一条尤其重要：别人 clone 下来的仓库不该继承你
        机器的 ctx 数字。

        note: 装不下显存等原因说明（degraded / MoE 混合形态时由 measure_ctx
              给），写进 ctx_measured_note[同键] 供 UI 卡片展示；正常全驻
              测量传空串，同时清掉该键的旧说明（旧结论作废）。
        mode: MoE 混合卸载模式。"cpu" = 该档以后每次加载都要带 --cpu-moe
              （测量验证过的形态，普通加载必须保持一致，否则测得的窗口
              换套卸载行为不一定还成立）；空串 = 清除该键旧记录。
              ⚠️ 2026-10-20 起是**回滚路径**：扫描全失败才会写 "cpu"。
        fit:  扫描选出的 fit 余量（MiB）。非空 → 写进 fit_target[同键]，
              该档以后每次加载都带 `-fitt <fit>`（同理：测量验证过的形态）。
              None → 清掉该键旧记录（退回 --cpu-moe 时）—— 必须与
              mode 在同一次写入里落盘，否则会出现"窗口是用 fit 测的、
              记录里却没有 fit_target"，下次加载退回 --cpu-moe，测出的
              窗口立刻不成立（方案 §6.1 坑 4）。
              两条记录互斥：有 fit_target 时 _start_inner 绝不再传 --cpu-moe。
        写盘后打一行侦探日志 + 回读校验（WB 2026-10-16 要求）：下次真重测
        时能当场抓住"写了没生效/被并发覆盖"的肇事者。
        """
        try:
            user = dict(load_config())
            key = KaiwuManager._ctx_cache_key(cfg, model)
            m = dict(user.get("ctx_measured") or {})
            m[key] = int(ctx)
            user["ctx_measured"] = m
            env = dict(user.get("ctx_measured_env") or {})
            env[key] = self._vram_fingerprint(cfg, model)
            user["ctx_measured_env"] = env
            notes = dict(user.get("ctx_measured_note") or {})
            if note:
                notes[key] = note
            elif key in notes:
                del notes[key]          # 又能全驻了 → 旧作废说明清掉
            user["ctx_measured_note"] = notes
            modes = dict(user.get("moe_offload") or {})
            if mode == "cpu":
                modes[key] = "cpu"
            elif key in modes:
                del modes[key]          # 不再需要混合卸载 → 旧记录作废
            user["moe_offload"] = modes
            # fit_target 与 ctx/指纹**同一次写入**（分开写会留下"窗口用 fit
            # 测的、记录里没 fit"的错配，见 docstring）。指纹复用同一份
            # ctx_measured_env：fit 的最优值同样只跟显卡/显存/KV/二进制/
            # 伴生挂载有关，不另设一套。
            fts = dict(user.get("fit_target") or {})
            if fit is not None:
                fts[key] = int(fit)
            elif key in fts:
                del fts[key]            # 退回 --cpu-moe → 旧 fit 档作废
            user["fit_target"] = fts
            save_config(user)
            print(f"[测量] 已写盘 {key} = {int(ctx)}"
                  + (f"（说明：{note}）" if note else "")
                  + (f"（fit 余量：{int(fit)} MiB）" if fit is not None else "")
                  + (f"（MoE 卸载：--cpu-moe）" if mode == "cpu" else ""))
            # 回读校验：落盘值 ≠ 期望值 → 当场警告（写入被吞 / 被并发覆盖）
            try:
                back = dict(load_config())
                got = (back.get("ctx_measured") or {}).get(key)
                got_fit = (back.get("fit_target") or {}).get(key)
                want_fit = int(fit) if fit is not None else None
                if got != int(ctx):
                    print(f"[测量] ⚠ 回读校验失败：期望 {key}={int(ctx)}，"
                          f"磁盘实为 {got!r}")
                elif got_fit != want_fit:
                    print(f"[测量] ⚠ 回读校验失败：期望 fit_target[{key}]="
                          f"{want_fit!r}，磁盘实为 {got_fit!r}")
            except Exception as e:
                print(f"[测量] 回读校验跳过（{e}）")
        except Exception as e:
            print(f"[测量] 写入缓存失败（不影响本次运行）：{e}")

    def _vram_fingerprint(self, cfg: dict, model: str = "") -> dict:
        """当前推理环境的指纹：显卡名 + 总显存 + KV 量化档 + llama-server 版本
        + 伴生挂载状态（视觉塔 / MTP 草稿实际挂没挂）。

        这几项任一变化，测出来的 ctx 就不再适用于当前环境。伴生挂载单列
        一项（2026-10-17）：视觉塔 ≈1.2G、MTP 草稿 ≈0.46G 显存，且
        llama.cpp fit 阶段量不到草稿 —— 开关前后测得的窗口会差一截。
        MoE 混合卸载把窗口拉到 188K 后显存余量更紧，这项更不能缺：不然
        用户开着草稿重载，拿到的就是"没草稿时"测的大窗口，直接起不来。

        ⚠️ 结果按 cfg 内容缓存：nvidia-smi + `llama-server --version` 两个子进程
        各要 ~0.3s，而 _ctx_cache_valid 每次加载模型都会问一次。不缓存的话，
        每换一个模型就白等 0.3~0.6s。环境在同一进程生命周期内不会变（换卡要
        重启控制台才有新配置生效），所以缓存是安全的。缓存键含挂载状态
        （不同模型的挂载不同，见下）。
        """
        # 实际挂载状态：按模型的 attach_overrides → 全局开关 → 伴生文件能否
        # 解析到。用户关了开关、或目录里没有塔/头，都算"没挂"。
        vis = dr = False
        if model:
            try:
                ov = (cfg.get("attach_overrides") or {}).get(model) or {}
                if ov.get("vision", cfg.get("attach_vision", True)):
                    vis = bool(self._resolve_mmproj(cfg, model))
                if ov.get("draft", cfg.get("attach_draft", True)):
                    dr = bool(self._resolve_draft(cfg, model))
            except Exception:
                pass
        attach = f"v{int(vis)}/d{int(dr)}"
        # 预填批量：compute buffer 随它涨（ub 2048 ≈ +700 MiB），显存账目
        # 变了测得的窗口就不成立，必须并进指纹。
        batch = f"{int(cfg.get('prefill_batch') or 0)}/{int(cfg.get('prefill_ubatch') or 0)}"
        cache_key = (str(cfg.get("cache_type_k") or ""), str(cfg.get("cache_type_v") or ""),
                     str(cfg.get("llama_server_exe") or ""), attach, batch)
        cached = KaiwuManager._fp_cache.get(cache_key)
        if cached is not None:
            return cached

        name, total = "", 0
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=8,
                creationflags=subprocess.CREATE_NO_WINDOW)
            if r.returncode == 0 and r.stdout.strip():
                first = r.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first.split(",")]
                name, total = parts[0], int(parts[1])
        except Exception:
            pass
        ver = ""
        try:
            ls = resolve_llama_server(cfg)
            if ls:
                v = subprocess.run([ls, "--version"], capture_output=True, text=True,
                                   timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
                blob = (v.stdout or "") + (v.stderr or "")
                m = re.search(r"version:\s*(\S+)|b(\d{3,6})", blob)
                if m:
                    ver = next(g for g in m.groups() if g)
        except Exception:
            pass
        fp = {
            "gpu": name, "vram_mib": total,
            "kv": f"{cfg.get('cache_type_k') or 'f16'}/{cfg.get('cache_type_v') or 'f16'}",
            "llama": ver,
            # 伴生挂载状态（视觉塔/MTP 草稿实际挂没挂）—— 加载期参数，
            # 开关后显存预算差 ~1.7G，测得的窗口不再成立
            "attach": attach,
            # 预填批量（-b/-ub）—— compute buffer 随它涨，同上
            "batch": batch,
            # 判据版本：判据修订过要 +1，旧记录在新判据下不可信，强制重测一次。
            #   1 = 只按层数（最初版）
            #   2 = 层数在 GPU 且权重不 mmap 到内存（2026-10-16）
            #   3 = 稠密沿用 2，MoE 改为"层数在 GPU + 实测 tok/s 达门槛"，
            #       未全驻的 MoE 优先 --cpu-moe 混合卸载（2026-10-17）
            #       —— 旧判据下 gemma-4-26B-A4B 被冤判"装不下"记 32768，
            #       必须作废重测。
            #   4 = 稠密 / MoE 门槛分离：稠密混合档要过 ctx_dense_min_tps
            #       （默认 30，2026-10-17 用户拍板"小稠密至少 30，不拿半速
            #       糊弄"）—— 旧门槛 5 下测出的稠密大窗口（如 Ornith
            #       114688 @12.5 tok/s）在新标准下不可信，作废重测。
            "crit": "4",
        }
        KaiwuManager._fp_cache[cache_key] = fp
        return fp

    def _ctx_cache_valid(self, cfg: dict, key: str, model: str = "") -> bool:
        """ctx_measured[key] 在当前环境下是否仍然可信。
        key = 「模型@量化档」（见 _ctx_cache_key），也兼容旧平名模型名。
        model: 模型名（非空时指纹含伴生挂载状态，见 _vram_fingerprint）。
                调用方一般都有模型名，传上 —— 不传则挂载项按"没挂"算，
                开了塔/头的记录会被判失效重测（保守方向，不会用错窗口）。

        没有任何指纹记录（老缓存 / 别人 clone 的仓库）→ 不可信，必须重测。
        """
        env = (cfg.get("ctx_measured_env") or {}).get(key)
        if not isinstance(env, dict):
            return False
        cur = self._vram_fingerprint(cfg, model)
        for k in ("gpu", "vram_mib", "kv", "llama", "attach", "batch", "crit"):
            if str(env.get(k, "")) != str(cur.get(k, "")):
                return False
        return True

    @staticmethod
    def _model_matches(served: list[str], want: str) -> bool:
        """模型 id 可能带路径/标签后缀，宽松匹配。"""
        if not want:
            return True
        want_l = want.lower()
        for s in served:
            sl = s.lower()
            if want_l == sl or want_l in sl or sl in want_l:
                return True
            if Path_stem(want_l) and Path_stem(want_l) in sl:
                return True
        return False


def Path_stem(name: str) -> str:
    """'D:/x/foo.gguf' → 'foo'；只剥已知权重扩展名。

    ⚠️ 不能用 rsplit('.',1) 这类「见点就切」的写法：模型名常带版本点号
    （qwen3.5-4b-ud 会被截成 qwen3），而 'qwen3' 是所有 qwen 模型路径的
    公共子串 —— 曾经据此让 4B/9B 两张卡片同时亮「运行中」绿灯。
    （ctx 查表那几处早已记录过同一陷阱，见 _target_ctx 的注释。）"""
    s = name.replace("\\", "/").split("/")[-1]
    low = s.lower()
    for ext in (".gguf", ".bin", ".safetensors", ".pth", ".ckpt"):
        if low.endswith(ext):
            return s[: -len(ext)]
    return s
