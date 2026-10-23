"""_test_pick_front.py — 「添加模型文件夹」弹窗可见性修复的规格测试。

2026-10-17 用户实测（整合包 kaiwu.exe 里）：点「＋ 添加模型文件夹」，
Windows 文件夹选择框要么排在最大化的浏览器后面，要么整个没弹出来，
界面还跳假「已取消」——连踩三次，用户以为功能坏了。

根因两条，本测试各锁一条：
  1. 对话框是 kaiwu 进程弹的，后台进程直接置前台会被 Windows 前台锁
     拒绝；且浏览器最大化时对话框天然排在后面 → 用户看不见 → 按 Esc
     → Show 返回 ERROR_CANCELLED。修法：_bring_console_front() 在 Show
     前还原最小化 + 挂输入线程绕前台锁 + 置前台 + 任务栏闪烁兜底，
     并把拿到的 HWND 当 owner 传给 Show。
  2. 旧代码 `except OSError: return None` 把「对话框没起来」和「用户
     取消」混为一谈 —— 前台锁拦下的失败也返回取消类 HRESULT，用户看到
     的就是一个哪都没弹出的「已取消」。修法：_show_cancelled() 只认
     ERROR_CANCELLED(1223)；其他 HRESULT 抛出去走降级 / 报真实错误。

真机验证留给手动：双击 dist-bundle/kaiwu.exe → 点添加模型文件夹 →
控制台窗口应被带到前台 + 任务栏闪烁 + 对话框出现在最上层。
"""

import subprocess as _sp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import local_fs
from app.local_fs import FsError

ok = fail = 0


def check(cond: bool, msg: str) -> None:
    global ok, fail
    print(("  [OK] " if cond else "  [FAIL] ") + msg)
    if cond:
        ok += 1
    else:
        fail += 1


def _oserr(winerror):
    e = OSError("mock")
    e.winerror = winerror
    return e


print("== [1] _arm_dialog_pin 契约（存在 / 不崩 / 无返回值）==")
check(callable(getattr(local_fs, "_arm_dialog_pin", None)),
      "函数存在")
try:
    ret = local_fs._arm_dialog_pin()
    check(ret is None, f"无返回值（实际 {ret!r}；不再返回 owner）")
except Exception as e:
    check(False, f"调用抛异常：{type(e).__name__}: {e}")

print("== [1b] 假 ctypes 注入：前台三连 + 看门狗钉顶全链路 ==")
# 2026-10-17 事故复盘：bcab18c 的 _bring_console_front 第一句用 ctypes，
# 但模块顶层没 import → NameError 被 except 吞掉 → 恒返回 0 → 置前逻辑
# 100% 静默失效，而沙箱里"返回 0"跟"没有控制台"长得一样，测试照过。
# 沙箱里 AllocConsole 不可靠（建不出真控制台），改用假 ctypes 注入
# sys.modules——函数内 import ctypes 会拿到它，从而验证整条链路真的跑了：
#   ① 拿到控制台 HWND → IsIconic/SetForegroundWindow/BringWindowToTop/闪烁
#   ② 看门狗线程起来 → 轮询到 #32770 对话框 → SetWindowPos(-1) 钉最顶
# 老 bug（缺 import）下这个测试全红——假货就在 sys.modules 里也救不了它。
class _Fn:
    """可调用的假 API——支持 .argtypes 赋值（真 bound method 不支持）。"""

    def __init__(self, fn):
        self.fn = fn
        self.argtypes = None

    def __call__(self, *a, **k):
        return self.fn(*a, **k)


class _Holder:
    def __init__(self, v=0):
        self.value = v


CONSOLE_HWND, DLG_HWND, FG_HWND, MY_PID = 4242, 777, 999, 1234
_rec: list[tuple] = []


class _Buf:
    value = ""


class _FakeU32:
    def __init__(self):
        self._enum_n = 0
        # 会被赋 .argtypes 的方法必须用 _Fn 包（bound method 不能加属性）
        self.SetWindowPos = _Fn(self._set_window_pos)
        self.GetWindowThreadProcessId = _Fn(self._get_wtid)

    def IsIconic(self, h):
        return False

    def ShowWindow(self, h, c):
        _rec.append(("ShowWindow", h, c))
        return 1

    def GetForegroundWindow(self):
        return FG_HWND

    def _get_wtid(self, h, pid):
        if pid is not None:
            pid.value = MY_PID          # 全是本进程的窗口
        return 555                      # 前台线程 id ≠ 本线程

    def AttachThreadInput(self, a, b, attach):
        _rec.append(("AttachThreadInput", a, b, attach))
        return 1                        # 非零 = 挂接成功

    def SetForegroundWindow(self, h):
        _rec.append(("SetForegroundWindow", h))
        return 1

    def SwitchToThisWindow(self, h, alt):
        _rec.append(("SwitchToThisWindow", h))
        return None

    def BringWindowToTop(self, h):
        _rec.append(("BringWindowToTop", h))
        return 1

    def FlashWindowEx(self, fi):
        _rec.append(("FlashWindowEx", getattr(fi, "hwnd", "?")))
        return 1

    def IsWindowVisible(self, h):
        return h == DLG_HWND

    def GetWindowLongW(self, h, idx):
        return 0                        # 模拟"永远不在顶层"→ 反复补钉

    def GetClassNameW(self, h, buf, n):
        buf.value = "#32770" if h == DLG_HWND else ""
        return 64

    def EnumWindows(self, cb, lparam):
        self._enum_n += 1
        if self._enum_n >= 2:           # 第 1 次=快照（对话框还不存在）
            cb(DLG_HWND, lparam)        # 之后=看门狗：对话框出现了
        return True

    def _set_window_pos(self, h, after, x, y, w, ht, flags):
        _rec.append(("SetWindowPos", h, getattr(after, "value", after)))
        return 1


class _FakeK32:
    def GetConsoleWindow(self):
        return CONSOLE_HWND

    def GetCurrentProcessId(self):
        return MY_PID

    def GetCurrentThreadId(self):
        return 999                      # ≠ 前台线程 555 → 走 AttachThreadInput


class _FakeWindll:
    user32 = _FakeU32()
    kernel32 = _FakeK32()


class _FakeCtypes:
    windll = _FakeWindll()

    @staticmethod
    def WINFUNCTYPE(*a):
        return lambda fn: fn

    @staticmethod
    def sizeof(x):
        return 16

    @staticmethod
    def byref(x):
        return x

    @staticmethod
    def create_unicode_buffer(n):
        return _Buf()

    @staticmethod
    def c_void_p(v=0):
        return _Holder(v)

    @staticmethod
    def c_ulong(v=0):
        return _Holder(v)

    @staticmethod
    def c_bool(v=False):
        return _Holder(v)

    @staticmethod
    def c_int(v=0):
        return _Holder(v)

    @staticmethod
    def c_uint(v=0):
        return _Holder(v)

    class Structure:
        def __init__(self, *a, **k):
            for i, v in enumerate(a):
                setattr(self, ("cbSize", "hwnd", "dwFlags", "uCount",
                               "dwTimeout")[i] if i < 5 else f"f{i}", v)

    @staticmethod
    def POINTER(t):
        return t


_real_ctypes = sys.modules.get("ctypes")
sys.modules["ctypes"] = _FakeCtypes          # type: ignore[assignment]
# [1] 已经起过一个真看门狗（单例标志还占着），这里强制清掉模拟干净进程；
# 那个真看门狗用真 ctypes 跑，看不到假窗口，不会干扰本段断言。
local_fs._PIN_STATE["live"] = False
try:
    local_fs._arm_dialog_pin()
    # 看门狗在子线程里跑，给它最多 3 秒钉上
    import time as _t
    for _ in range(60):
        if any(r[0] == "SetWindowPos" for r in _rec):
            break
        _t.sleep(0.05)
finally:
    if _real_ctypes is not None:
        sys.modules["ctypes"] = _real_ctypes
    else:
        del sys.modules["ctypes"]
    local_fs._PIN_STATE["live"] = False

check(not any(r[0] == "SetForegroundWindow"
              and getattr(r[1], "value", r[1]) == CONSOLE_HWND for r in _rec)
      and not any(r[0] == "BringWindowToTop"
                  and getattr(r[1], "value", r[1]) == CONSOLE_HWND
                  for r in _rec)
      and not any(r[0] == "FlashWindowEx" for r in _rec)
      and not any(r[0] == "IsIconic" for r in _rec),
      f"控制台完全不被碰（owner 毒药已除；实际调用：{sorted({r[0] for r in _rec})}）")
check(any(r[0] == "SetWindowPos"
          and getattr(r[1], "value", r[1]) == DLG_HWND and r[2] == -1
          for r in _rec),
      f"看门狗把 #32770 对话框钉成 TOPMOST(-1)（实际：{[r for r in _rec if r[0] == 'SetWindowPos']}）")
check(any(r[0] == "SetForegroundWindow"
          and getattr(r[1], "value", r[1]) == DLG_HWND
          for r in _rec)
      and any(r[0] == "AttachThreadInput" for r in _rec),
      f"TOPMOST 被窗口过程拒时走置前兜底（挂前台线程 + 置前台的是**对话框**）")

print("== [2] _show_cancelled：只有 ERROR_CANCELLED 算取消 ==")
# ⚠️ 2026-10-17 二修：ctypes 抛的 OSError.winerror 是**完整有符号 HRESULT**
# （-2147023673），不是十进制 1223 也不是正数 0x800704C7。第一版按那三个
# 值比对，真实取消永远对不上 → 用户按取消被误判"对话框没弹出来"→ source
# 模式再弹一个 Tk 框、frozen 模式直接报错。现在统一 & 0xFFFFFFFF 归一化。
for we in (-2147023673, 0x800704C7):
    check(local_fs._show_cancelled(_oserr(we)) is True,
          f"winerror={we}（ERROR_CANCELLED 的有符号/无符号写法）→ 算取消")
for we in (5, 0x80070005, -2147024891, 0, None, 1224, 1223, -2147021801):
    check(local_fs._show_cancelled(_oserr(we)) is False,
          f"winerror={we}（前台锁拒绝/旧版错值/其他失败）→ 不算取消")

print("== [3] pick_folder 边界行为（fake _pick_native 驱动）==")


class _FakeSub:
    """拦住 Tk 降级路径里的 subprocess.run，避免真弹框卡住测试。
    带 TimeoutExpired 属性—— pick_folder 的 except 子句要引用它。"""

    TimeoutExpired = _sp.TimeoutExpired

    def run(self, *a, **k):
        raise OSError("mock: 备用对话框也起不来")


def _run_pick(native_behavior, frozen=False):
    """驱动 pick_folder。native_behavior：返回值 / 要抛的异常。
    返回 (kind, value)：ok=正常返回 / fserr=FsError / other=其他异常。"""
    def fake(title):
        if isinstance(native_behavior, Exception):
            raise native_behavior
        return native_behavior
    local_fs._pick_native = fake
    local_fs.subprocess = _FakeSub()
    saved_frozen = getattr(sys, "frozen", None)
    if frozen:
        sys.frozen = True
    try:
        return ("ok", local_fs.pick_folder(timeout=1))
    except FsError as e:
        return ("fserr", str(e))
    except Exception as e:
        return ("other", f"{type(e).__name__}: {e}")
    finally:
        local_fs._pick_native = orig_native
        local_fs.subprocess = _sp
        if saved_frozen is None:
            if hasattr(sys, "frozen"):
                del sys.frozen
        else:
            sys.frozen = saved_frozen


orig_native = local_fs._pick_native

kind, val = _run_pick(None)
check(kind == "ok" and val is None,
      "native 说取消（返回 None）→ pick_folder 返回 None，不进降级")

kind, val = _run_pick("/fake/models/dir")
check(kind == "ok" and val == "/fake/models/dir",
      "正常选中 → 原样返回路径，不进降级")

kind, val = _run_pick(_oserr(5))
check(kind == "fserr" and "无法弹出选择框" in val,
      "native 抛真失败 → 走降级；降级也失败时报 FsError，不假冒取消")

kind, val = _run_pick(OSError("mock 无 winerror"))
check(kind == "fserr", "OSError 无 winerror → 同样按真失败走降级")

kind, val = _run_pick(NotImplementedError("无 ctypes"))
check(kind == "fserr", "NotImplementedError → 走降级（降级失败报真实错）")

kind, val = _run_pick(None, frozen=True)
check(kind == "ok" and val is None,
      "冻结模式（整合包 exe）下取消同样返回 None")

kind, val = _run_pick(_oserr(5), frozen=True)
check(kind == "fserr" and "没弹出来" in val and "重启" in val,
      "冻结模式真失败 → FsError 文案说人话（恢复窗口可见 + 重启建议）")

print("== [4] 源码结构：无 owner + 看门狗（含置前兜底）+ 不碰控制台 ==")
src = open("app/local_fs.py", encoding="utf-8").read()
pick_src = src.split("def _pick_native")[1]
front_src = src.split("def _arm_dialog_pin")[1].split("def _show_cancelled")[0]
check("_arm_dialog_pin()" in pick_src, "_pick_native 里调用了 _arm_dialog_pin")
check("(nul, owner or None)" in pick_src,
      "Show 传 TOPMOST owner（我们自己建的、会泵消息的窗口；不是控制台）")
check("_show_cancelled(e)" in pick_src, "except 分支走 _show_cancelled 分辨")
check("threading" in front_src and "#32770" in front_src
      and "SetWindowPos" in front_src,
      "TOPMOST 看门狗在场（线程轮询 #32770 对话框类）")
check("AttachThreadInput(" in front_src,
      "置前兜底在场（TOPMOST 被窗口过程拒时挂前台线程激活对话框）")
check("done.add" in front_src,
      "每个窗口只折腾一次（不 20Hz 刷屏重试）")
check("import ctypes" in front_src.split('"""')[2],
      "_arm_dialog_pin 内部有 import ctypes（bcab18c 静默失效的死因）")
check("u32.IsWindowVisible(" not in front_src,
      "看门狗不要求 IsWindowVisible（owner 链不可见时它会一直撒谎）")
check("GetConsoleWindow(" not in front_src,
      "已不再触碰控制台窗口（owner 毒药源头）")

print(f"\n===== {'全部通过' if fail == 0 else '有失败'}：{ok} 过 / {fail} 挂 =====")
sys.exit(0 if fail == 0 else 1)
