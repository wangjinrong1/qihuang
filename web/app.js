const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ---------- 极简 Markdown ---------- */
function md(src) {
  const lines = esc(src).split("\n"); const out = []; let list = false;
  const inline = t => t.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  const closeList = () => { if (list) { out.push("</ul>"); list = false; } };
  for (const l of lines) {
    let m;
    if ((m = l.match(/^(#{1,3})\s+(.*)/))) { closeList(); out.push(`<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`); }
    else if ((m = l.match(/^\s*[-*]\s+(.*)/))) { if (!list) { out.push("<ul>"); list = true; } out.push(`<li>${inline(m[1])}</li>`); }
    else if ((m = l.match(/^&gt;\s?(.*)/))) { closeList(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); }
    else if (!l.trim()) closeList();
    else { closeList(); out.push(`<p>${inline(l)}</p>`); }
  }
  closeList(); return out.join("");
}

/* ---------- SSE (POST) ---------- */
async function sse(url, init, onEvent) {
  const res = await fetch(url, init);
  if (!res.ok) { let t = res.statusText; try { t = (await res.json()).detail || t; } catch {} throw new Error(t); }
  const reader = res.body.getReader(), dec = new TextDecoder(); let buf = "";
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    buf += dec.decode(value, { stream: true });
    let i; while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
      if (chunk.startsWith("data: ")) onEvent(JSON.parse(chunk.slice(6)));
    }
  }
}

/* ---------- 标签页 ---------- */
$$("#nav button").forEach(b => b.onclick = () => {
  $$("#nav button").forEach(x => x.classList.toggle("on", x === b));
  $$(".tab").forEach(t => t.classList.toggle("on", t.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "atlas") { const f = $("#atlasFrame"); if (!f.src) f.src = f.dataset.src; }
  if (b.dataset.tab === "kb" && !kbLoaded) loadKB();
});

/* ---------- 状态 ---------- */
let LLM = null, WEB = null;
const MODE_NAME = { api: "云端 API", local: "本地中医模型", mock: "演示模式" };
function updateChip() {
  if (!LLM) return;
  const name = LLM.switching ? "切换中…" : LLM.mode === "api" ? `云端 · ${LLM.api_model}` : MODE_NAME[LLM.mode] || LLM.mode;
  $("#modelChip").textContent = `模型：${name} ⚙`;
  const prov = WEB?.providers?.join(" → ") || "无";
  $("#webLabel").title = `开启后，会把症状/证型关键词（不含个人信息）发送到搜索引擎（${prov}），用网页摘要补充模型知识；不开启则完全离线。`;
}
async function refreshStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    const L = s.llm, dot = $(".dot"); LLM = L; WEB = s.web; updateChip();
    dot.className = "dot";
    let text;
    if (L.state === "loading") { dot.classList.add("demo"); text = "模型加载中…"; setTimeout(refreshStatus, 3000); }
    else if (L.mode === "mock") { dot.classList.add("demo"); text = L.error ? "模型加载失败（演示模式）" : "演示模式（未配置模型）"; }
    else { dot.classList.add("ok"); text = L.mode === "api" ? `云端 API 已就绪 · ${L.base_model}` : `中医大模型已就绪 · ${L.device === "cuda" ? "GPU" : "CPU"}`; }
    $("#statusText").textContent = text;
    $("#status").title = `模型：${L.base_model}\n适配器：${L.adapter || "未配置"}\n设备：${L.device || "-"}\n加载耗时：${L.load_seconds ?? "-"}s\n舌象：${s.tongue.mode}　药材图片：${s.herb.mode}　OCR：${s.ocr ? "可用" : "未安装"}\n图谱：${s.kg.syndromes}证型 / ${s.kg.formulas}方剂 / ${s.kg.herbs}药材\n资料块：${s.rag.chunks}` + (L.error ? `\n\n加载失败：${L.error}` : "");
    $("#kbStats").textContent = `${s.kg.symptoms} 症状 · ${s.kg.syndromes} 证型 · ${s.kg.formulas} 方剂 · ${s.kg.herbs} 药材 · ${s.kg.incompatible_pairs} 条配伍禁忌 · 古籍 ${s.rag.documents} 部 / ${s.rag.chunks} 段`;
  } catch { $("#statusText").textContent = "后端未连接"; }
}
refreshStatus();

/* ---------- 智能问诊 ---------- */
let history = [];
const box = $("#chatBox");
function addMsg(role, html) {
  $(".hello", box)?.remove();
  const d = document.createElement("div"); d.className = "msg " + role;
  d.innerHTML = `<div class="avatar">${role === "user" ? "我" : "岐"}</div><div class="bubble">${html}</div>`;
  box.appendChild(d); box.scrollTop = box.scrollHeight; return $(".bubble", d);
}
const webToggle = $("#webToggle");
webToggle.checked = localStorage.getItem("webSearch") === "1";
webToggle.onchange = () => localStorage.setItem("webSearch", webToggle.checked ? "1" : "0");
const prog = $("#chatProgress");
function showProgress(p) {
  // 收集阶段：告诉用户“已了解什么、还想了解什么”，并允许随时要求直接给建议
  if (!p || p.mode !== "collect") { prog.hidden = true; return; }
  const known = p.known.length ? `已了解：${p.known.join("、")}` : "正在了解您的情况";
  const unk = p.unknown.length ? `　还想了解：${p.unknown.slice(0, 3).join("、")}` : "";
  $("#chatProgressText").textContent = known + unk;
  $("#chatConclude").hidden = !p.can_conclude;
  prog.hidden = false;
}
async function send(text, force = false) {
  text = text.trim(); if (!text) return;
  prog.hidden = true;
  $("#chatInput").value = ""; $("#chatSend").disabled = true;
  addMsg("user", esc(text));
  const bub = addMsg("ai", ""); bub.classList.add("cursor"); let ans = "", refs = "";
  try {
    await sse("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, history, force_conclude: force, web_search: webToggle.checked }) }, ev => {
      if (ev.type === "status") { bub.innerHTML = `<span class="muted">${esc(ev.text)}</span>`; }
      else if (ev.type === "meta") {
        showProgress(ev.progress);
        const t = [...ev.symptoms.map(s => `症状·${s}`), ...ev.syndromes.map(s => `证型·${s}`), ...ev.sources.map(s => `资料·${s}`)];
        const links = (ev.web || []).map(w => `<a class="tag web" href="${esc(w.url)}" target="_blank" rel="noopener noreferrer" title="${esc(w.url)}">网络·${esc(w.title.slice(0, 18))}</a>`);
        const note = ev.web_error ? `<div class="muted small">联网检索没有成功（${esc(ev.web_error)}），本次只用了本地资料。</div>` : "";
        refs = (t.length || links.length) ? `<div class="refs">${t.map(x => `<span class="tag">${esc(x)}</span>`).join("")}${links.join("")}</div>${note}` : note;
      } else if (ev.type === "token") { ans += ev.text; bub.innerHTML = `<div class="md">${md(ans)}</div>`; box.scrollTop = box.scrollHeight; }
      else if (ev.type === "error") ans += `\n\n（出错：${ev.message}）`;
    });
  } catch (e) { ans += `（请求失败：${e.message}）`; }
  bub.classList.remove("cursor"); bub.innerHTML = `<div class="md">${md(ans)}</div>${refs}`;
  // 加载提示/报错不是真实回答，不放进对话历史
  if (!/^[（【](模型正在加载|出错|请求失败|生成失败|API 调用失败|演示模式)/.test(ans.trim()))
    history.push({ role: "user", content: text }, { role: "assistant", content: ans });
  $("#chatSend").disabled = false; $("#chatInput").focus();
}
$("#chatSend").onclick = () => send($("#chatInput").value);
$("#chatConclude").onclick = () => send("信息就这些了，请直接给我辨证和调护建议", true);
$("#chatInput").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(e.target.value); } });
$$(".quick button").forEach(b => b.onclick = () => send(b.textContent));
$("#chatClear").onclick = () => location.reload();

/* ---------- 辨证报告 ---------- */
const STEPS = [["collect", "信息"], ["tongue", "舌象"], ["kg", "辨证"], ["rag", "检索"], ["section", "分析"], ["report", "报告"]];
const imgInput = $("#imgInput");
imgInput.onchange = () => {
  const f = imgInput.files[0], p = $("#imgPrev");
  if (!f) { p.hidden = true; $("#dropText").hidden = false; return; }
  p.src = URL.createObjectURL(f); p.hidden = false; $("#dropText").hidden = true;
};

$("#consultForm").onsubmit = async e => {
  e.preventDefault();
  const fd = new FormData(e.target), patient = {};
  ["name", "gender", "age", "complaint", "duration", "history"].forEach(k => patient[k] = (fd.get(k) || "").toString().trim());
  const body = new FormData(); body.append("patient", JSON.stringify(patient));
  if (imgInput.files[0]) body.append("image", imgInput.files[0]);

  const R = $("#result"); $("#consultGo").disabled = true; $("#consultGo").textContent = "辨证中…";
  R.innerHTML = `<div class="card"><div class="steps" id="steps">${STEPS.map(([k, l]) => `<span class="step" data-k="${k}">${l}</span>`).join("")}</div></div>
    <div class="card conflict" id="cFlag" hidden></div>
    <div class="card" id="cInfo" hidden></div>
    <div class="card" id="cSym" hidden><h4>识别到的症状</h4><div class="chips"></div></div>
    <div class="card" id="cTongue" hidden><h4>舌象</h4><div></div></div>
    <div class="card" id="cSyn" hidden><h4>图谱辨证</h4><div></div></div>
    <div class="card analysis" id="cAna" hidden><h4>分析与建议</h4><div></div></div>
    <div class="card" id="cRep" hidden><h4>完整报告</h4><div class="report md"></div><div class="actions"><button class="ghost" id="dl">下载 Markdown</button><button class="ghost" id="pr">打印</button></div></div>`;
  const stepEls = Object.fromEntries($$(".step", R).map(s => [s.dataset.k, s]));
  const order = STEPS.map(x => x[0]);
  const mark = k => { const i = order.indexOf(k); order.forEach((kk, j) => stepEls[kk].className = "step" + (j < i ? " done" : j === i ? " on" : "")); };
  const secs = {};

  try {
    await sse("/api/consult", { method: "POST", body }, ev => {
      switch (ev.type) {
        case "stage": mark(ev.stage); break;
        case "symptoms": { const c = $("#cSym"); c.hidden = false;
          $(".chips", c).innerHTML = ev.symptoms.length ? ev.symptoms.map(s => `<span class="tag">${esc(s)}</span>`).join("") : `<span class="muted">未匹配到图谱收录的症状，可换更具体的表述</span>`; break; }
        case "tongue": { const c = $("#cTongue"); c.hidden = false; const t = ev.tongue;
          $("div", c).innerHTML = `<span class="tag">${esc(t.label)}</span> ${t.demo ? '<span class="tag warn">演示结果</span>' : `置信度 ${(t.confidence * 100).toFixed(0)}%`}<p>${esc(t.note)}</p>`; break; }
        case "syndromes": renderSyn(ev); break;
        case "section_start": { const c = $("#cAna"); c.hidden = false;
          const s = document.createElement("div"); s.innerHTML = `<h5>${esc(ev.title)}</h5><p class="cursor"></p>`; $("div", c).appendChild(s); secs[ev.title] = { el: $("p", s), t: "" }; break; }
        case "token": { const s = secs[ev.title]; s.t += ev.text; s.el.innerHTML = md(s.t); s.el.className = "cursor"; break; }
        case "redflags": { const c = $("#cFlag"); c.hidden = false;
          c.innerHTML = `<h4>需要尽快就医的信号</h4>` + ev.items.map(x => `<p>🚨 ${esc(x)}</p>`).join(""); break; }
        case "insufficient": {
          $$(".step", R).forEach(x => x.className = "step");
          const c = $("#cInfo"); c.hidden = false;
          const tpl = ["寒热：", "口渴/口干：", "大小便：", "食欲/睡眠：", "出汗：", "咳嗽咳痰：", "局部特点："].join("\n");
          c.innerHTML = `<h4>信息不足，暂不能生成辨证报告</h4>
            <p class="muted">依据不足时直接出报告容易编造内容。请补充下面的必备信息后再点“开始辨证”。</p>
            <h5>还需要补充（必备）</h5><ul>${ev.need.map(x => `<li>${esc(x)}</li>`).join("")}</ul>
            ${ev.suggest.length ? `<h5>建议补充</h5><ul>${ev.suggest.map(x => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
            <h5>已掌握</h5><div class="chips">${ev.have.map(h => `<span class="tag">${esc(h.item)}：${esc(h.detail)}</span>`).join("") || "<span class=\"muted\">暂无</span>"}</div>
            <div class="actions"><button class="ghost" id="fillTpl">在主诉后追加填写模板</button></div>`;
          $("#fillTpl").onclick = () => { const t = $("#consultForm [name=complaint]"); t.value = t.value.trim() + "\n" + tpl; t.focus(); }; break; }
        case "section_reset": { const s = secs[ev.title]; s.t = ""; s.el.innerHTML = ""; break; }
        case "section_replace": { const s = secs[ev.title]; s.t = ev.text; s.el.innerHTML = md(s.t); break; }
        case "report": { Object.values(secs).forEach(s => s.el.className = ""); $$(".step", R).forEach(s => s.className = "step done");
          const c = $("#cRep"); c.hidden = false; $(".report", c).innerHTML = md(ev.markdown);
          $("#dl").onclick = () => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([ev.markdown], { type: "text/markdown" })); a.download = "辨证报告.md"; a.click(); };
          $("#pr").onclick = () => { const w = window.open(""); w.document.write(`<meta charset=utf-8><title>辨证报告</title><body style="font-family:sans-serif;max-width:800px;margin:2em auto">${$(".report", c).innerHTML}`); w.print(); }; break; }
        case "error": R.insertAdjacentHTML("beforeend", `<div class="card conflict">出错：${esc(ev.message)}</div>`); break;
      }
    });
  } catch (err) { R.insertAdjacentHTML("beforeend", `<div class="card conflict">请求失败：${esc(err.message)}</div>`); }
  $("#consultGo").disabled = false; $("#consultGo").textContent = "开始辨证";
};

function formulaHTML(f) {
  const herbs = f.herbs.map(h => `<span class="herb" title="${esc([h.nature, h.meridian, h.flavor].filter(Boolean).join(" · "))}">${esc(h.name)}${h.dose_g ? `<small>${h.dose_g}g</small>` : ""}</span>`).join("");
  const conf = f.conflicts.map(c => `<div class="conflict">⚠️ 配伍禁忌（${esc(c.rule)}）：${esc(c.a)} ↔ ${esc(c.b)}，须由医师审核</div>`).join("");
  return `<details class="frm" open><summary>${esc(f.name)}</summary><div class="herbs">${herbs}</div>${conf}</details>`;
}
function renderSyn(ev) {
  const c = $("#cSyn"); c.hidden = false; let h = "";
  if (!ev.syndromes.length) h = `<p>图谱未匹配到证型，请补充更具体的症状（如寒热、舌脉、二便、饮食等）。</p>`;
  ev.syndromes.forEach(s => {
    h += `<div class="syn"><div class="syn-top"><b>${esc(s.name)}</b><div class="bar"><i style="width:${(s.score * 100).toFixed(0)}%"></i></div><span class="muted">匹配 ${(s.score * 100).toFixed(0)}%</span></div>
      <div class="chips" style="margin-top:6px">${s.matched_symptoms.map(x => `<span class="tag">${esc(x)}</span>`).join("")}</div>${s.formulas.map(formulaHTML).join("")}</div>`;
  });
  if (ev.direct_formulas?.length) h += `<h4 style="margin-top:14px">症状直达方剂（补充）</h4>` + ev.direct_formulas.map(formulaHTML).join("");
  $("div", c).innerHTML = h;
}

/* ---------- 知识库 ---------- */
let kbLoaded = false, kbData = { syn: [], frm: [] }, kbMode = "syn";
async function loadKB() {
  kbLoaded = true;
  [kbData.syn, kbData.frm] = await Promise.all([fetch("/api/kg/syndromes").then(r => r.json()), fetch("/api/kg/formulas").then(r => r.json())]);
  renderKB();
}
function renderKB() {
  if (kbMode === "book") return renderBooks();
  $("#kbSearch").placeholder = "搜索证型、方剂、药材…";
  const q = $("#kbSearch").value.trim();
  const items = kbMode === "syn"
    ? kbData.syn.filter(x => !q || x.name.includes(q) || x.formulas.some(f => f.includes(q))).map(x => `<div class="card kb-item syn" data-id="${esc(x.id)}" style="cursor:pointer"><b>${esc(x.name)}</b> <small class="muted">点击展开</small><p>典型症状 ${x.symptom_count} 项</p><p>方剂：${esc(x.formulas.join("、") || "—")}</p><div class="syn-detail" hidden></div></div>`)
    : kbData.frm.filter(x => !q || x.name.includes(q) || x.herbs.some(h => h.name.includes(q))).map(x => `<div class="card kb-item"><b>${esc(x.name)}</b><div class="herbs">${x.herbs.map(h => `<span class="herb">${esc(h.name)}${h.dose_g ? `<small>${h.dose_g}g</small>` : ""}</span>`).join("")}</div>${x.conflicts.length ? `<div class="conflict">含配伍禁忌</div>` : ""}</div>`);
  $("#kbList").innerHTML = items.join("") || `<p class="muted">无匹配结果</p>`;
  $$("#kbList .syn").forEach(el => el.onclick = async () => {
    const box = $(".syn-detail", el);
    if (!box.hidden) { box.hidden = true; return; }
    box.hidden = false;
    if (box.dataset.ok) return;
    box.innerHTML = `<p class="muted">加载中…</p>`;
    try {
      const r = await fetch(`/api/kg/syndrome/${encodeURIComponent(el.dataset.id)}`);
      if (!r.ok) throw new Error(r.statusText);
      const d = await r.json();
      box.innerHTML = `<h5>典型症状（${d.symptoms.length}）</h5><div class="chips">${d.symptoms.map(x => `<span class="tag">${esc(x)}</span>`).join("")}</div>`
        + `<h5>治疗方剂</h5>` + (d.formulas.map(formulaHTML).join("") || `<p class="muted">图谱中暂无方剂</p>`);
      box.dataset.ok = "1";
    } catch (e) { box.innerHTML = `<p class="muted">加载失败：${esc(e.message)}</p>`; }
  });
}
$("#kbSearch").oninput = () => { kbMode === "book" ? (clearTimeout(kbTimer), kbTimer = setTimeout(renderBooks, 300)) : renderKB(); };
$$("#kbSeg button").forEach(b => b.onclick = () => { kbMode = b.dataset.k; kbBook = ""; $("#kbSearch").value = ""; $$("#kbSeg button").forEach(x => x.classList.toggle("on", x === b)); renderKB(); });


/* ---------- 模型切换 ---------- */
const dlg = $("#modelDlg"), modelErr = $("#modelErr");
function syncDlg() {
  const mode = $('input[name="mode"]:checked')?.value;
  $("#apiFields").style.display = mode === "api" ? "" : "none";
  $("#localHint").style.display = mode === "local" ? "" : "none";
  $("#apiModelCustom").hidden = $("#apiModelSel").value !== "__custom";
}
function openModelDlg() {
  if (!LLM) return;
  modelErr.hidden = true;
  $(`input[name="mode"][value="${LLM.mode}"]`).checked = true;
  const known = [...$("#apiModelSel").options].some(o => o.value === LLM.api_model);
  $("#apiModelSel").value = known ? LLM.api_model : "__custom"; $("#apiModelCustom").value = known ? "" : LLM.api_model;
  $("#apiBase").value = LLM.api_base || ""; $("#apiKey").value = "";
  $("#apiKey").placeholder = LLM.api_key_hint ? `已设置（${LLM.api_key_hint}），留空沿用` : "sk-…";
  $("#localHint").textContent = LLM.local_configured ? "将在后台加载，加载期间继续使用当前模型。" : "还没有配置本地适配器（.env 的 LLM_ADAPTER_PATH），切换大概率会失败，失败时会保持当前模型。";
  syncDlg(); dlg.showModal();
}
$$('input[name="mode"]').forEach(r => r.onchange = syncDlg);
$("#apiModelSel").onchange = syncDlg;
$("#modelChip").onclick = $("#status").onclick = openModelDlg;
$("#modelCancel").onclick = () => dlg.close();
$("#modelForm").onsubmit = async e => {
  e.preventDefault();
  const mode = $('input[name="mode"]:checked').value, btn = $("#modelOk");
  const model = $("#apiModelSel").value === "__custom" ? $("#apiModelCustom").value.trim() : $("#apiModelSel").value;
  btn.disabled = true; btn.textContent = mode === "api" ? "测试连接中…" : "切换中…"; modelErr.hidden = true;
  try {
    const res = await fetch("/api/llm/switch", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, api_key: $("#apiKey").value, api_base: $("#apiBase").value, api_model: model }) });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    $("#apiKey").value = ""; dlg.close(); refreshStatus();
  } catch (err) { modelErr.textContent = "切换失败，已保持原模型：" + err.message; modelErr.hidden = false; }
  btn.disabled = false; btn.textContent = "测试并切换";
};


/* ---------- 古籍资料库（知识库 → 古籍资料） ---------- */
let kbBooks = null, kbTimer = null, kbBook = "";
function hl(text, q) {                      // 高亮查询里出现过的词（按 2 字切分）
  let t = esc(text);
  [...new Set(q.replace(/[\s，。、；,.]+/g, " ").split(" ").flatMap(w => w.length <= 2 ? [w] : [w, ...Array.from({ length: w.length - 1 }, (_, i) => w.slice(i, i + 2))]))]
    .filter(w => w.length >= 2).sort((a, b) => b.length - a.length).slice(0, 8)
    .forEach(w => { t = t.split(esc(w)).join(`<mark>${esc(w)}</mark>`); });
  return t;
}
async function renderBooks() {
  const q = $("#kbSearch").value.trim(), L = $("#kbList");
  $("#kbSearch").placeholder = kbBook ? `在《${kbBook}》里搜索，留空阅读原文…` : "在全部古籍里搜索症状、方名、穴位…（留空查看书目）";
  const bar = kbBook ? `<div class="kb-scope"><span>范围：<b>《${esc(kbBook)}》</b></span><a href="#" id="kbAll">取消，搜索全部</a></div>` : "";
  L.classList.toggle("reading", !!(q || kbBook));
  if (!q && !kbBook) {
    if (!kbBooks) kbBooks = await fetch("/api/corpus/books").then(r => r.json());
    L.classList.remove("reading");
    L.innerHTML = kbBooks.map(b => `<div class="card kb-item book" data-b="${esc(b.name)}" style="cursor:pointer"><b>${esc(b.name)}</b><p class="muted">${esc([b.dynasty, b.author].filter(Boolean).join(" · ") || "—")}　${b.chunks} 段 · ${b.size_kb >= 1024 ? (b.size_kb / 1024).toFixed(1) + " MB" : b.size_kb + " KB"}</p></div>`).join("");
    $$(".book", L).forEach(el => el.onclick = () => { kbBook = el.dataset.b; $("#kbSearch").value = ""; $("#kbSearch").focus(); renderBooks(); });
    return;
  }
  if (!q) { await readBook(0); return; }
  L.innerHTML = bar + `<p class="muted">检索中…</p>`; bindAll();
  try {
    const hits = await fetch(`/api/corpus/search?q=${encodeURIComponent(q)}&k=12&book=${encodeURIComponent(kbBook)}`).then(r => r.json());
    if (q !== $("#kbSearch").value.trim()) return;                 // 输入已变化，丢弃过期结果
    L.innerHTML = bar + (hits.length ? `<div class="reader">${hits.map(h => `<div class="reader-seg hit"><h5>${esc(h.source)}</h5><p>${hl(h.text, q)}</p></div>`).join("")}</div>` : `<p class="muted">没有找到相关段落，换个说法试试（如把“拉肚子”换成“泄泻”）。</p>`);
  } catch (e) { L.innerHTML = bar + `<p class="muted">检索失败：${esc(e.message)}</p>`; }
  bindAll();
  async function readBook(offset) {
    if (offset === 0) { L.innerHTML = bar + `<p class="muted">加载中…</p>`; bindAll(); }
    try {
      const r = await fetch(`/api/corpus/book?name=${encodeURIComponent(kbBook)}&offset=${offset}&limit=20`);
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
      const d = await r.json();
      if ($("#kbSearch").value.trim() || d.name !== kbBook) return;      // 已开始搜索或换了书
      const html = d.items.map((c, i) => `<div class="reader-seg">${c.title ? `<h5>${esc(c.title)}</h5>` : ""}<p>${esc(c.text)}</p></div>`).join("");
      const end = d.offset + d.items.length, more = end < d.total;
      const foot = `<span>已显示 ${end} / ${d.total} 段</span>${more ? `<button class="ghost" id="kbMore">继续加载</button>` : ""}`;
      if (offset === 0) L.innerHTML = bar + `<div id="bookBody" class="reader">${html}</div><div id="bookFoot" class="reader-foot">${foot}</div>`;
      else { $("#bookBody").insertAdjacentHTML("beforeend", html); $("#bookFoot").innerHTML = foot; }
      bindAll();
      const m = $("#kbMore"); if (m) m.onclick = e => { e.preventDefault(); readBook(end); };
    } catch (e) { L.innerHTML = bar + `<p class="muted">加载失败：${esc(e.message)}</p>`; bindAll(); }
  }
  function bindAll() { const a = $("#kbAll"); if (a) a.onclick = e => { e.preventDefault(); kbBook = ""; renderBooks(); }; }
}
