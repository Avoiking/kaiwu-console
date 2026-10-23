/* 锁住「稍后再说（不联网）」的行为（2026-10-17 用户实测反馈的 bug）：
   点了就必须当场关弹窗，不管网络好不好使。
   测试对象是从 app/webui.py 里抽出来的真实函数，不是手抄的副本——
   抄一份只会测出"抄得对"。
   跑法：node _test_key_skip.js */
const fs = require("fs");
const vm = require("vm");

const path = require("path");
const SRC = path.join(__dirname, "..", "app", "webui.py");
const src = fs.readFileSync(SRC, "utf8");
const s = src.indexOf("<script>") + 8;
const e = src.indexOf("</script>");
if (s < 8 || e < 0) { console.log("FAIL: 找不到 script 块"); process.exit(1); }
const js = src.slice(s, e);

let pass = 0, fail = 0;
function ok(cond, label) {
  if (cond) { pass++; console.log("  ✅ " + label); }
  else { fail++; console.log("  ❌ " + label); }
}

/* 从整页 JS 里抠出指定函数体（按花括号配对）。这些函数体内没有字符串/
   模板花括号，配对安全；真遇到了让测试炸出来，比静默截错强。 */
function extractFn(name) {
  const sig = "function " + name;
  let i = js.indexOf(sig);
  while (i > 0 && /[A-Za-z0-9_$.]/.test(js[i - 1])) {
    i = js.indexOf(sig, i + 1);   // 跳过 xxxFunction 这类名字里含它的
  }
  if (i < 0) throw new Error("源码里找不到函数 " + name);
  if (js.slice(i - 6, i) === "async ") i -= 6;   // 别把 async 前缀切丢了
  let depth = 0, started = false, j = js.indexOf("{", i);
  if (j < 0) throw new Error("函数 " + name + " 找不到函数体");
  for (let k = j; k < js.length; k++) {
    if (js[k] === "{") { depth++; started = true; }
    else if (js[k] === "}") { depth--; if (started && depth === 0) { j = k + 1; break; } }
  }
  if (depth !== 0) throw new Error("函数 " + name + " 花括号没配对");
  return js.slice(i, j);
}

/* ---- 搭一个假 DOM + 假 fetch 的沙箱 ---- */
function makeSandbox(fetchImpl) {
  const els = {};
  const fake = (id) => els[id] || (els[id] = {
    id, hidden: false, textContent: "", value: "",
    style: {}, title: "", dataset: {},
    classList: { add() {}, remove() {} },
  });
  const toasts = [];
  const sandbox = { $: fake, fetch: fetchImpl, toast: (m) => toasts.push(m), console };
  const ctx = vm.createContext(sandbox);
  vm.runInContext("let webOn=false, webQuota=null;", ctx);
  vm.runInContext([
    extractFn("hideKeyGuide"),
    extractFn("paintWebBtn"),
    extractFn("saveKey"),
    extractFn("skipKey"),
  ].join("\n\n"), ctx);
  return { ctx, els: fake, toasts };
}
const tick = () => new Promise((r) => setTimeout(r, 0));

async function main() {
console.log("== [1]  wiring：keySkip 必须接到 skipKey（不再走 saveKey）==");
const wi = js.indexOf('$("keySkip").onclick=');
const wiring = wi >= 0 ? js.slice(wi, js.indexOf(";", wi)) : "(没找到)";
ok(wi >= 0 && wiring.endsWith("skipKey"), "实际接线：" + wiring);

console.log("== [2]  fetch 成功：立即关窗 + 后台落盘 + toast ==");
{
  let called = null;
  const { ctx, els, toasts } = makeSandbox(async (url, opt) => {
    called = { url, body: JSON.parse(opt.body) };
    return { json: async () => ({ ok: true, has_key: false, enabled: false,
                                  key_skipped: true, used: 0, limit: 1000 }) };
  });
  vm.runInContext("skipKey()", ctx);
  ok(els("keyMask").hidden === true, "调用当帧弹窗就没了（同步生效，不等 fetch）");
  ok(els("btnWeb").textContent === "联网", "联网按钮回到未开启状态");
  await tick(); await tick();
  ok(called && called.url === "/api/web/key" && called.body.key === "",
     "后台仍把跳过态 POST 给 /api/web/key（key 为空）");
  ok(toasts.includes("已跳过，暂时不联网"), "toast：已跳过，暂时不联网");
  ok(vm.runInContext("webOn", ctx) === false, "webOn 保持 false");
  ok(vm.runInContext("webQuota && webQuota.key_skipped", ctx) === true,
     "webQuota 已更新（key_skipped=true，下次启动不自动弹）");
}

console.log("== [3]  fetch 失败（用户截图里的 Failed to fetch）：这是本次修的 bug ==");
{
  const { ctx, els, toasts } = makeSandbox(async () => {
    throw new TypeError("Failed to fetch");
  });
  vm.runInContext("skipKey()", ctx);
  ok(els("keyMask").hidden === true,
     "fetch 挂了弹窗也立即消失（旧代码：死守着不走，只能点叉叉）");
  await tick(); await tick();
  ok(els("keyMsg").textContent === "",
     "不在弹窗里留错误信息（弹窗都关了，留了也没人看）");
  ok(toasts.length === 0, "不假报'已跳过'——落盘没成就是没成，下次启动再引导");
  ok(vm.runInContext("webOn", ctx) === false, "webOn 仍正确保持 false");
}

console.log("== [4]  对照：saveKey(\"\") 失败时仍留在弹窗（保存路径要能重试，刻意不同）==");
{
  const { ctx, els } = makeSandbox(async () => {
    throw new TypeError("Failed to fetch");
  });
  await vm.runInContext("saveKey(\"\")", ctx).catch(() => {});
  ok(els("keyMask").hidden === false, "保存失败 → 弹窗留着 + 错误可见，用户可重试");
  ok(/保存失败/.test(els("keyMsg").textContent), "错误信息写进 keyMsg");
}

console.log("== [5]  回归：保存有效 key 的路径没被碰坏 ==");
{
  const { ctx, els, toasts } = makeSandbox(async (url, opt) => {
    const k = JSON.parse(opt.body).key;
    if (k && !/^tvly-/.test(k)) throw new Error("bad key");
    return { json: async () => ({ ok: true, has_key: !!k, enabled: !!k,
                                  used: 0, limit: 1000, remaining: 1000 }) };
  });
  els("keyInput").value = "tvly-abc123";
  vm.runInContext('saveKey($("keyInput").value)', ctx);
  await tick(); await tick();
  ok(els("keyMask").hidden === true, "有效 key → 弹窗关闭");
  ok(toasts.includes("已保存，联网可用"), "toast：已保存，联网可用");
  ok(vm.runInContext("webOn", ctx) === true, "webOn 置 true");
  els("keyInput").value = "abc123";
  await vm.runInContext('saveKey($("keyInput").value)', ctx).catch(() => {});
  ok(/tvly- 开头/.test(els("keyMsg").textContent), "非 tvly- 前缀仍给提醒");
}

console.log("\n===== " + (fail === 0 ? "全部通过" : "有失败") +
            "：" + pass + " 过 / " + fail + " 挂 =====");
process.exit(fail === 0 ? 0 : 1);
}

main();
