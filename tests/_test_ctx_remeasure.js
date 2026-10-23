/* 上下文弹窗（openCtxPicker）DOM 冒烟测试
 *
 * 目的：验证新增的「重新测量」选项在三种卡片状态下都正确出现/隐藏，
 * 且提交给后端的 payload 符合预期（remeasure 标记、手填值、默认模式）。
 *
 * 跑法：NODE_PATH=<workspace>/node_modules node _test_ctx_remeasure.js
 */
const fs = require("fs");
const { JSDOM } = require("jsdom");

const SRC = fs.readFileSync(__dirname + "/app/webui.py", "utf8");
const i = SRC.indexOf('PAGE = r"""') + 'PAGE = r"""'.length;
const j = SRC.indexOf('"""', i);
const PAGE = SRC.slice(i, j);

/* —— 三种待测状态 —— */
const STATES = {
  measured: { name: "Ornith-1.5-9B", measured: 32768, override: 0, stale: false, auto: true, variant: "", active: "" },
  override: { name: "Ornith-1.5-9B", measured: 0, override: 32768, stale: false, auto: true, variant: "", active: "" },
  fresh:    { name: "Brand-New-Model", measured: 0, override: 0, stale: false, auto: true, variant: "", active: "" },
  stale:    { name: "Ornith-1.5-9B", measured: 32768, override: 0, stale: true, auto: true, variant: "", active: "" },
};

let pass = 0, fail = 0;
function ok(cond, label) {
  if (cond) { pass++; console.log("  ✓ " + label); }
  else { fail++; console.log("  ✗ " + label); }
}

function boot() {
  // dangerously：让 <script> 在页面上下文里正常执行，函数声明才会挂上 window。
  // 必须给 url：脚本一上来就摸 localStorage，opaque origin 下会 SecurityError。
  const dom = new JSDOM(PAGE, {
    runScripts: "dangerously",
    pretendToBeVisual: true,
    url: "http://127.0.0.1:8080/",
    beforeParse(w) {
      // 掐掉自动轮询/会话加载（它们会 fetch 我们没桩的端点），其余脚本不动
      w.fetch = async (url, opt) => {
        const u = String(url);
        if (u.includes("/api/use")) {
          w.__last_use = JSON.parse(opt.body);
          return { json: async () => ({ ok: true, model: "x", ctx: 49152,
                                         remeasured: !!w.__last_use.remeasure }) };
        }
        return { json: async () => ({ ok: true, sessions: [], models: [], on: false }) };
      };
      w.__last_use = null;
    },
  });
  const w = dom.window;
  // 等脚本里的 refresh() 首轮跑完（它是异步的）
  return w;
}

function popup(w, m) {
  w.openCtxPicker(m, null);
  const p = w.document.querySelector("body > div[style*='position: fixed']");
  if (!p) throw new Error("弹窗没渲染出来");
  return p;
}

/* 弹窗里三个选项块的标题，按顺序
 * 注意：mkOpt 的结构是 o > div[flex] > (span.ctxdot + div > (title + desc))，
 * 用 querySelector("div div") 会连嵌套的描述行一起抓 —— 必须精确取
 * 「带 .ctxdot 的那一层」里的第 2 个子元素的第 1 个子元素。 */
function titles(p) {
  return [...p.querySelectorAll(".ctxdot")]
    .map(dot => dot.parentElement.children[1].children[0].textContent.trim());
}

function pickByTitle(p, t) {
  const dot = [...p.querySelectorAll(".ctxdot")].find(d =>
    d.parentElement.children[1].children[0].textContent.trim() === t);
  return dot ? dot.parentElement.parentElement : null;
}

function submit(w, p) {
  const go = [...p.querySelectorAll("button")].find(b => b.textContent === "加载");
  go.click();
  return w.__last_use;
}

/* ============ 1. 有测量记录 → 三个选项都在 ============ */
console.log("\n[1] 有测量记录（32768）");
{
  const w = boot();
  const p = popup(w, STATES.measured);
  const t = titles(p);
  ok(t.length === 3, "三个选项都渲染：" + t.join(" / "));
  ok(t.includes("重新测量"), "「重新测量」出现");

  const r = pickByTitle(p, "重新测量");
  ok(/丢弃现有的\s*32K/.test(r.textContent), "文案提到旧值 32K");
  ok(/换显卡|环境已变/.test(r.textContent), "文案提到换显卡/环境已变");

  // 默认态：已有记录 → 默认手动填写（沿用旧行为）
  r.click();
  const body = submit(w, p);
  ok(body.remeasure === true, "提交带 remeasure=true");
  ok(body.ctx === null, "重测时不带手填值");
  ok(body.name === "Ornith-1.5-9B", "带对模型名");
}

/* ============ 2. 手动指定过 → 重测要说明会撤销手填 ============ */
console.log("\n[2] 已手动指定 32K");
{
  const w = boot();
  const p = popup(w, STATES.override);
  const t = titles(p);
  ok(t.length === 3, "三个选项都渲染：" + t.join(" / "));
  const r = pickByTitle(p, "重新测量");
  ok(/先撤销这个手填值/.test(r.textContent), "明确提示会撤销手填值");
  r.click();
  const body = submit(w, p);
  ok(body.remeasure === true, "提交带 remeasure=true");
  ok(body.ctx === null, "不带旧的手填值");
}

/* ============ 3. 从没测过 → 不显示「重新测量」 ============ */
console.log("\n[3] 全新模型（无任何记录）");
{
  const w = boot();
  const p = popup(w, STATES.fresh);
  const t = titles(p);
  ok(t.length === 2, "只有两个选项：" + t.join(" / "));
  ok(!t.includes("重新测量"), "「重新测量」不出现（没有旧记录可弃）");
  ok(t[0] === "自动测量", "默认第一项是自动测量");
}

/* ============ 4. 指纹过期 → 仍可手动重测 ============ */
console.log("\n[4] 记录已过期（换过硬件）");
{
  const w = boot();
  const p = popup(w, STATES.stale);
  const t = titles(p);
  ok(t.includes("重新测量"), "「重新测量」出现");
  const r = pickByTitle(p, "重新测量");
  ok(/本次加载本来就会自动重测/.test(r.textContent), "说明自动重测也会发生");
}

/* ============ 5. 手填路径仍然正常 ============ */
console.log("\n[5] 选「手动填写」提交数字");
{
  const w = boot();
  const p = popup(w, STATES.measured);
  pickByTitle(p, "手动填写").click();
  const inp = p.querySelector("input[type=number]");
  ok(!!inp, "手填输入框出现");
  inp.value = "65536";
  const body = submit(w, p);
  ok(body.ctx === 65536, "带上手填值 65536");
  ok(body.remeasure === false, "remeasure 为 false");
}

/* ============ 6. 重新测量 + 手填互斥（后端也要兜住） ============ */
console.log("\n[6] 边界：remeasure 与 ctx 同时出现 → 后端以 remeasure 为准");
{
  // 这条由 Python 侧 _use 保证（remeasure 分支先 clear_model_ctx，忽略 ctx）
  const w = boot();
  const p = popup(w, STATES.measured);
  pickByTitle(p, "重新测量").click();
  const body = submit(w, p);
  ok(body.remeasure === true && body.ctx === null, "前端就不会同时发两者：" + JSON.stringify(body));
}

console.log(`\n===== ${pass} 通过 / ${fail} 失败 =====`);
process.exit(fail ? 1 : 0);