/* 药方识别 & 中药识别（复用 app.js 的 $, esc, sse, md） */

/* ---------- 药方识别 ---------- */
const DEMO_RX = "柴胡12g 黄芩9g 姜半夏9g 党参10g 炙甘草6g 生姜3片 大枣4枚";
$("#rxDemo").onclick = () => { $("#rxText").value = DEMO_RX; };
$("#rxImg").onchange = e => {
  const f = e.target.files[0], p = $("#rxPrev");
  p.hidden = !f; $("#rxDropText").hidden = !!f; if (f) p.src = URL.createObjectURL(f);
};

$("#rxForm").onsubmit = async e => {
  e.preventDefault();
  const body = new FormData(); body.append("text", $("#rxText").value);
  const img = $("#rxImg").files[0]; if (img) body.append("image", img);
  const R = $("#rxResult"); $("#rxGo").disabled = true; $("#rxGo").textContent = "识别中…";
  R.innerHTML = `<div class="card" id="rxStage" hidden><span class="tag"></span></div>
    <div class="card" id="rxHerbs" hidden><h4>药材清单</h4><div class="tbl"></div></div>
    <div class="card" id="rxMatch" hidden><h4>相近方剂（按药材重合度）</h4><div></div></div>
    <div class="card analysis" id="rxAna" hidden><h4>解读</h4><div class="md cursor"></div></div>`;
  let txt = "";
  try {
    await sse("/api/prescription", { method: "POST", body }, ev => {
      switch (ev.type) {
        case "stage": { const c = $("#rxStage"); c.hidden = false; $(".tag", c).textContent = ev.label + "…"; break; }
        case "text": $("#rxText").value = ev.text; break;       // OCR 结果回填，可手动修正后重新识别
        case "parsed": renderParsed(ev); break;
        case "token": { const c = $("#rxAna"); c.hidden = false; txt += ev.text; const d = $(".md", c); d.innerHTML = md(txt); break; }
        case "done": $("#rxStage").hidden = true; $("#rxAna .md").classList.remove("cursor"); break;
        case "error": R.insertAdjacentHTML("beforeend", `<div class="card conflict">${esc(ev.message)}</div>`); $("#rxStage").hidden = true; break;
      }
    });
  } catch (err) { R.insertAdjacentHTML("beforeend", `<div class="card conflict">请求失败：${esc(err.message)}</div>`); }
  $("#rxGo").disabled = false; $("#rxGo").textContent = "识别药方";
};

function renderParsed(ev) {
  const bad = new Set(ev.conflicts.flatMap(c => [c.a, c.b]));
  const rows = ev.herbs.map(h => {
    const dose = h.dose == null ? "<span class='muted'>未写</span>" :
      `${h.dose}${esc(h.unit)}${h.dose_approx ? `<small class="muted"> ≈${h.dose_g}g</small>` : ""}${h.unit_assumed ? `<small class="muted"> 默认g</small>` : ""}`;
    return `<tr class="${bad.has(h.name) ? "bad" : ""}"><td><b class="herb-link" style="cursor:pointer" data-h="${esc(h.base)}">${esc(h.name)}</b></td><td>${dose}</td>
      <td>${esc(h.nature)}${h.flavor ? " · " + esc(h.flavor) : ""}</td><td>${esc(h.meridian)}</td><td>${esc(h.note)}</td></tr>`;
  }).join("");
  const c = $("#rxHerbs"); c.hidden = false;
  $(".tbl", c).innerHTML = `<table class="rx"><tr><th>药名</th><th>剂量</th><th>性 · 味</th><th>归经</th><th>备注</th></tr>${rows}</table>` +
    ev.conflicts.map(x => `<div class="conflict">⚠️ 配伍禁忌（${esc(x.rule)}）：${esc(x.a)} 与 ${esc(x.b)}，须由执业医师审核</div>`).join("") +
    (ev.unknown.length ? `<div class="muted" style="margin-top:8px">未能识别：${ev.unknown.map(esc).join("、")}（可能是错字或图谱未收录）</div>` : "");
  $$(".herb-link", c).forEach(el => el.onclick = () => { openHerb(el.dataset.h); });
  if (ev.matches.length) {
    const m = $("#rxMatch"); m.hidden = false;
    $("div", m).innerHTML = ev.matches.map(x => `<div class="match"><b>${esc(x.name)}</b><div class="bar"><i style="width:${x.similarity * 100}%"></i></div><span class="muted">${(x.similarity * 100).toFixed(0)}%</span>
      <span class="muted" style="font-size:12px">缺：${esc(x.missing.slice(0, 6).join("、") || "—")}　多：${esc(x.extra.slice(0, 6).join("、") || "—")}</span></div>`).join("") +
      `<p class="muted" style="font-size:12px;margin:8px 0 0">仅按药材重合度比对图谱中的方剂；图谱方剂数据来自病例汇总，组成不一定等同经典方，结果仅作参考。</p>`;
  }
}

/* ---------- 中药识别 / 查询 ---------- */
$("#herbImg").onchange = e => {
  const f = e.target.files[0], p = $("#herbPrev");
  p.hidden = !f; $("#herbDropText").hidden = !!f; if (f) p.src = URL.createObjectURL(f);
};
$("#herbGo").onclick = async () => {
  const f = $("#herbImg").files[0]; if (!f) { alert("请先选择药材图片"); return; }
  const body = new FormData(); body.append("image", f);
  const btn = $("#herbGo"); btn.disabled = true; btn.textContent = "识别中…";
  try {
    const r = await fetch("/api/herb/recognize", { method: "POST", body });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const d = await r.json();
    $("#herbResult").innerHTML = `<div class="card"><h4>识别候选 ${d.demo ? '<span class="tag warn">演示结果，未接入识别模型</span>' : ""}</h4>
      <div class="chips">${d.candidates.map(c => `<span class="tag clickable" data-h="${esc(c.name)}">${esc(c.name)}${d.demo ? "" : " " + (c.confidence * 100).toFixed(0) + "%"}</span>`).join("")}</div>
      ${d.demo ? '<p class="muted" style="font-size:12px">演示模式的候选是随机给出的，仅用于查看界面流程。配置 HERB_WEIGHTS 与 HERB_LABELS 后才是真实识别。</p>' : ""}</div><div id="herbCard"></div>`;
    $$("#herbResult .tag.clickable").forEach(t => t.onclick = () => showHerb(t.dataset.h, "#herbCard"));
    if (d.candidates[0]) showHerb(d.candidates[0].name, "#herbCard");
  } catch (e) { $("#herbResult").innerHTML = `<div class="card conflict">识别失败：${esc(e.message)}</div>`; }
  btn.disabled = false; btn.textContent = "识别药材";
};

let sugTimer;
$("#herbQ").oninput = e => {
  clearTimeout(sugTimer);
  sugTimer = setTimeout(async () => {
    const q = e.target.value.trim(); if (!q) { $("#herbSug").innerHTML = ""; return; }
    const names = await (await fetch("/api/herb/search?q=" + encodeURIComponent(q))).json();
    $("#herbSug").innerHTML = names.map(n => `<span class="tag clickable">${esc(n)}</span>`).join("") || `<span class="muted">未收录</span>`;
    $$("#herbSug .tag.clickable").forEach(t => t.onclick = () => openHerb(t.textContent));
  }, 200);
};

function openHerb(name) {           // 从任意页面跳到“中药识别”页查看详情
  $('#nav button[data-tab="herb"]').click();
  if (!$("#herbCard")) $("#herbResult").innerHTML = `<div id="herbCard"></div>`;
  showHerb(name, "#herbCard");
}
async function showHerb(name, sel) {
  const box = $(sel);
  const r = await fetch("/api/herb/" + encodeURIComponent(name));
  if (!r.ok) { box.innerHTML = `<div class="card">图谱中未收录「${esc(name)}」</div>`; return; }
  const d = await r.json();
  box.innerHTML = `<div class="card"><h4 style="font-size:20px;font-family:'Songti SC',serif">${esc(d.name)}</h4>
    <div class="kv"><span>性</span><span>${esc(d.nature) || "—"}</span><span>味</span><span>${esc(d.flavor) || "—"}</span><span>归经</span><span>${esc(d.meridian) || "—"}</span></div>
    <h4 style="margin-top:14px">出现在这些方剂中（${d.formulas.length}）</h4>
    <div class="chips">${d.formulas.map(f => `<span class="tag">${esc(f.name)}${f.dose_g ? " " + f.dose_g + "g" : ""}</span>`).join("") || "<span class='muted'>图谱中暂无</span>"}</div>
    ${d.taboo.length ? `<h4 style="margin-top:14px">配伍禁忌</h4><div class="chips">${d.taboo.map(t => `<span class="tag warn">${esc(t.herb)}（${esc(t.rule)}）</span>`).join("")}</div>` : ""}
    <p class="muted" style="font-size:12px;margin-top:12px">性味归经数据来自图谱词表并经术语规范化，仅供学习参考，使用前请核对权威药典。</p></div>`;
}
