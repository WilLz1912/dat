const POS = "#2a9d8f", NEG = "#e76f51", ACC = "#3d5a80", GRID = "#e3e6ec";
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (x) => (x * 100).toFixed(0) + "%";
Chart.defaults.font.family = '"Segoe UI", system-ui, sans-serif';
Chart.defaults.color = "#5b6476";
Chart.defaults.maintainAspectRatio = false;
Chart.defaults.animation = false;
const charts = {};
function draw(id, cfg) {
  charts[id]?.destroy();
  charts[id] = new Chart(document.getElementById(id), cfg);
}
async function api(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

/* ---------- tabs ---------- */
const loaded = {};
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("active", p.id === b.dataset.tab));
  if (!loaded[b.dataset.tab]) { loaded[b.dataset.tab] = true; ({ overview: loadOverview, reco: loadReco }[b.dataset.tab] || (() => {}))(); }
}));

/* ---------- 1. phân tích 1 review ---------- */
const EXAMPLES = [
  ["Absolutely love this game. The art style is beautiful, the soundtrack is amazing and I've sunk 100 hours without getting bored. 10/10", "true", 100],
  ["Crashes every 10 minutes since the last patch. Lost my save twice. Devs don't respond. Refunded.", "false", 1.5],
  ["Best game I've played this year, great story and the combat feels super smooth. Highly recommend to everyone!", "false", 40],
  ["good good good good good good good good", "true", 3],
  ["Oh great, another update that breaks everything. Thanks devs, really appreciate paying for a beta test.", "true", 300],
];
document.querySelectorAll("[data-ex]").forEach((b) => b.addEventListener("click", () => {
  const [t, v, p] = EXAMPLES[+b.dataset.ex];
  $("#s-text").value = t; $("#s-vote").value = v; $("#s-pt").value = p;
  runSingle();
}));
$("#s-run").addEventListener("click", runSingle);

async function runSingle() {
  const text = $("#s-text").value.trim();
  if (!text) return;
  const vote = $("#s-vote").value, pt = $("#s-pt").value;
  $("#s-result").innerHTML = '<div class="loading">Đang phân tích…</div>';
  try {
    const r = await api("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, voted_up: vote === "" ? null : vote === "true", playtime: pt === "" ? null : +pt }),
    });
    const pos = r.sentiment === "positive";
    const helpLevel = r.p_helpful >= .6 ? "cao" : r.p_helpful >= .4 ? "trung bình" : "thấp";
    $("#s-result").innerHTML = `
      <h2>Kết quả</h2>
      <div class="verdict"><span class="badge ${pos ? "pos" : "neg"}">${pos ? "KHEN 👍" : "CHÊ 👎"}</span>
        <span>Model ${r.confidence} (${pct(pos ? r.p_positive : 1 - r.p_positive)})</span></div>
      <div class="meter"><div class="lbl"><span>Mức tích cực của nội dung</span><b>${pct(r.p_positive)}</b></div>
        <div class="bar"><div style="width:${r.p_positive * 100}%;background:${pos ? POS : NEG}"></div></div></div>
      <div class="meter"><div class="lbl"><span>Khả năng được vote "hữu ích" (sau ~30 ngày)</span><b>${pct(r.p_helpful)} · ${helpLevel}</b></div>
        <div class="bar"><div style="width:${r.p_helpful * 100}%;background:${ACC}"></div></div></div>
      ${r.suspicious
        ? `<div class="alert"><b>⚠ Review đáng ngờ</b><ul>${r.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>`
        : r.note ? `<div class="alert" style="background:var(--warnbg);border-color:#f0d9a8;color:var(--warn)">Lưu ý: ${esc(r.note)}.</div>`
        : `<div class="ok">✓ Không thấy dấu hiệu bất thường${r.voted_up === null ? " (chọn nút Recommend để kiểm tra mâu thuẫn)" : ""}.</div>`}
      <div class="use"><b>Ứng dụng:</b><ul>
        <li>Nền tảng: review đáng ngờ → ẩn / đưa xuống cuối, không tính vào điểm game.</li>
        <li>Review khả năng hữu ích cao → đẩy lên đầu trang game ngay, không đợi người khác vote.</li>
        <li>Nhà phát triển: review chê → chuyển cho đội hỗ trợ / sửa lỗi.</li></ul></div>`;
  } catch (e) { $("#s-result").innerHTML = `<div class="alert">${esc(e.message)}</div>`; }
}

/* ---------- 2. hàng loạt ---------- */
let batchRows = [], batchName = "";
api("/api/games").then((gs) => {
  $("#b-game").innerHTML = gs.map((g) => `<option>${esc(g)}</option>`).join("");
  $("#b-game").value = gs.includes("Counter-Strike 2") ? "Counter-Strike 2" : gs[0];
});
$("#b-run-game").addEventListener("click", () => {
  const g = $("#b-game").value;
  runBatch(() => api("/api/batch/sample?game=" + encodeURIComponent(g)), g);
});
$("#b-file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  f.text().then((txt) => runBatch(() => api("/api/batch", { method: "POST", headers: { "Content-Type": "text/plain" }, body: txt }), f.name));
});

async function runBatch(fetcher, name) {
  $("#b-loading").classList.remove("hidden"); $("#b-out").classList.add("hidden");
  try {
    const res = await fetcher();
    batchRows = res.rows; batchName = name;
    renderBatch(res.summary);
  } catch (e) { alert(e.message); }
  $("#b-loading").classList.add("hidden");
}

function renderBatch(s) {
  $("#b-out").classList.remove("hidden");
  const kpi = (v, k) => `<div class="kpi"><div class="v">${v}</div><div class="k">${k}</div></div>`;
  $("#b-kpis").innerHTML = kpi(s.n, "review đã chấm") + kpi(pct(s.n_positive / s.n), "khen")
    + kpi(pct(s.n_negative / s.n), "chê") + kpi(s.n_suspicious, "review đáng ngờ") + kpi(s.n_uncertain, "model phân vân (review hỗn hợp)");
  draw("b-pie", { type: "doughnut", data: { labels: ["Khen", "Chê"], datasets: [{ data: [s.n_positive, s.n_negative], backgroundColor: [POS, NEG] }] },
    options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } } });
  draw("b-pt", { type: "bar", data: { labels: s.by_playtime.map((b) => [b.bin, "n=" + b.n]),
    datasets: [{ label: "% chê", data: s.by_playtime.map((b) => b.neg_pct), backgroundColor: NEG }] },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false } },
      scales: { y: { max: 100, grid: { color: GRID } }, x: { grid: { display: false } } } } });
  draw("b-kw", { type: "bar", data: { labels: s.keywords_neg.map((k) => k.term),
    datasets: [{ label: "số review chê có từ này", data: s.keywords_neg.map((k) => k.n), backgroundColor: NEG }] },
    options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false } },
      scales: { x: { grid: { color: GRID } }, y: { grid: { display: false }, ticks: { autoSkip: false } } } } });
  setFilter("all");
}

document.querySelectorAll("#b-filter button").forEach((b) => b.addEventListener("click", () => setFilter(b.dataset.f)));
function setFilter(f) {
  document.querySelectorAll("#b-filter button").forEach((b) => b.classList.toggle("on", b.dataset.f === f));
  let rows = batchRows.map((r, i) => ({ ...r, i: i + 1 }));
  if (f === "sus") rows = rows.filter((r) => r.suspicious);
  if (f === "neg") rows = rows.filter((r) => r.sentiment === "negative").sort((a, b) => a.p_positive - b.p_positive);
  if (f === "help") rows = [...rows].sort((a, b) => b.p_helpful - a.p_helpful).slice(0, 30);
  const vote = (v) => v === null ? "–" : v ? "👍" : "👎";
  $("#b-table").innerHTML = `<tr><th>#</th><th>Review</th><th>Model</th><th>Recommend</th><th>Giờ chơi</th><th>Hữu ích</th></tr>`
    + (rows.length ? rows.slice(0, 500).map((r) => `<tr>
      <td>${r.i}</td>
      <td class="txt">${esc(r.text_clean.slice(0, 400))}${r.text_clean.length > 400 ? "…" : ""}
        ${r.suspicious ? `<div class="reason">⚠ ${r.reasons.map(esc).join(" · ")}</div>` : ""}</td>
      <td><span class="pill ${r.sentiment === "positive" ? "pos" : "neg"}">${r.sentiment === "positive" ? "khen" : "chê"} ${pct(r.p_positive)}</span></td>
      <td>${vote(r.voted_up)}</td>
      <td>${r.playtime === null ? "–" : r.playtime.toFixed(1) + "h"}</td>
      <td>${pct(r.p_helpful)}</td></tr>`).join("")
      : `<tr><td colspan="6" class="empty">Không có review nào.</td></tr>`);
}

$("#b-dl").addEventListener("click", () => {
  const head = ["text_clean", "sentiment", "p_positive", "p_helpful", "voted_up", "playtime", "suspicious", "reasons"];
  const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const csv = [head.join(",")].concat(batchRows.map((r) => head.map((h) => q(h === "reasons" ? r.reasons.join("; ") : r[h])).join(","))).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["﻿" + csv], { type: "text/csv" }));
  a.download = batchName.replace(/\.csv$/i, "") + "_scored.csv";
  a.click();
});

/* ---------- 3. tổng quan ---------- */
async function loadOverview() {
  const o = await api("/api/overview");
  const kpi = (v, k) => `<div class="kpi"><div class="v">${v}</div><div class="k">${k}</div></div>`;
  $("#o-kpis").innerHTML = kpi(o.n_reviews.toLocaleString("vi"), "review sau làm sạch") + kpi(o.n_games, "game")
    + kpi(o.pos_pct + "%", "review khen") + kpi(o.n_mismatch, "review mâu thuẫn nội dung–nhãn") + kpi("0,89", "ROC-AUC model trong app");
  const sel = [...o.games].sort((a, b) => b.mismatch_pct - a.mismatch_pct).slice(0, 15);
  draw("o-games", { type: "bar", data: { labels: sel.map((x) => x.game_name),
    datasets: [{ label: "% review mâu thuẫn", data: sel.map((x) => x.mismatch_pct), backgroundColor: "#c0392b" }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } },
      scales: { x: { grid: { color: GRID } }, y: { grid: { display: false }, ticks: { autoSkip: false, font: { size: 11 } } } } } });
  draw("o-pt", { type: "bar", data: { labels: o.by_playtime.map((b) => b.bin),
    datasets: [{ label: "% khen", data: o.by_playtime.map((b) => b.pos_pct), backgroundColor: o.by_playtime.map((b) => b.pos_pct < 40 ? NEG : POS) }] },
    options: { plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: (c) => "n = " + o.by_playtime[c.dataIndex].n } } },
      scales: { y: { max: 100, grid: { color: GRID } }, x: { grid: { display: false } } } } });
  draw("o-metrics", { type: "bar", data: { labels: o.metrics.map((m) => m.model),
    datasets: [{ label: "A — game đã biết", data: o.metrics.map((m) => m.A), backgroundColor: ACC },
               { label: "B — game mới", data: o.metrics.map((m) => m.B), backgroundColor: "#98c1d9" }] },
    options: { indexAxis: "y", plugins: { legend: { position: "bottom" } },
      scales: { x: { min: .6, max: 1, grid: { color: GRID } }, y: { grid: { display: false }, ticks: { font: { size: 11 } } } } } });
  $("#o-sus").innerHTML = o.suspicious_examples.map((x) => `<div class="sus-item">
    <div class="meta">${esc(x.game)} · người viết bấm <b>${x.recommend ? "Recommend 👍" : "Không recommend 👎"}</b> · độ lệch ${x.score}</div>
    ${esc(x.text)}</div>`).join("");
}

/* ---------- 4. gợi ý ---------- */
let groups, network, nodes, genreColor;
async function loadReco() {
  groups = await api("/api/groups");
  const games = groups.games.filter((g) => g.is_game).sort((a, b) => a.game_name.localeCompare(b.game_name));
  $("#r-game").innerHTML = '<option value="">—</option>' + games.map((g) => `<option value="${g.app_id}">${esc(g.game_name)}</option>`).join("");
  $("#r-user").innerHTML = '<option value="">—</option>' + groups.users.map((u) =>
    `<option value="${u.user_id}">${u.user_id} · ${esc(u.game_name)} ${u.voted_up ? "👍" : "👎"}</option>`).join("");
  $("#r-user").addEventListener("change", () => {
    const u = groups.users.find((x) => x.user_id === $("#r-user").value);
    if (!u) return;
    $("#r-game").value = u.app_id; $("#r-liked").value = String(u.voted_up);
    runReco(u);
  });
  $("#r-game").addEventListener("change", () => { $("#r-user").value = ""; });
  $("#r-run").addEventListener("click", () => runReco(groups.users.find((x) => x.user_id === $("#r-user").value)));
  buildGraph();
  const bar = (id, rows, color) => draw(id, { type: "bar", data: { labels: rows.map((r) => `${r.name} (${r.n})`),
    datasets: [{ label: "% tích cực TB", data: rows.map((r) => r.pos), backgroundColor: color }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } },
      scales: { x: { min: 50, max: 100, grid: { color: GRID } }, y: { grid: { display: false }, ticks: { autoSkip: false, font: { size: 11 } } } } } });
  bar("r-genre", groups.by_genre, ACC);
  bar("r-pub", groups.by_publisher, "#98c1d9");
}

const PALETTE = ["#3d5a80", "#2a9d8f", "#e76f51", "#e9c46a", "#8e7dbe", "#4d908e", "#f4a261", "#577590", "#90be6d", "#c06c84", "#6d597a", "#43aa8b", "#b56576", "#277da1", "#f9844a"];
function buildGraph() {
  const genres = [...new Set(groups.games.map((g) => g.genre))];
  const color = genreColor = Object.fromEntries(genres.map((g, i) => [g, PALETTE[i % PALETTE.length]]));
  const N = [], E = [];
  genres.forEach((g) => N.push({ id: "g:" + g, label: g, shape: "box", color: color[g], font: { color: "#fff", size: 15 }, group0: "genre" }));
  groups.multi_publishers.forEach((p) => N.push({ id: "p:" + p, label: p, shape: "triangle", size: 12, color: "#9aa3b2", font: { size: 11 } }));
  groups.games.forEach((g) => {
    N.push({ id: g.app_id, label: g.game_name, shape: "dot", size: 6 + (g.game_pos_pct - 50) / 4,
             color: color[g.genre], font: { size: 11 }, title: `${g.game_name}\n${g.genre} · ${g.publisher || "?"} · ${g.game_pos_pct}% tích cực` });
    E.push({ from: g.app_id, to: "g:" + g.genre, color: { color: color[g.genre], opacity: .35 } });
    if (groups.multi_publishers.includes(g.publisher)) E.push({ from: g.app_id, to: "p:" + g.publisher, dashes: true, color: { color: "#9aa3b2", opacity: .6 } });
  });
  nodes = new vis.DataSet(N);
  network = new vis.Network($("#r-graph"), { nodes, edges: new vis.DataSet(E) }, {
    physics: { solver: "forceAtlas2Based", forceAtlas2Based: { gravitationalConstant: -40, springLength: 60 }, stabilization: { iterations: 250 } },
    interaction: { hover: true, tooltipDelay: 100 }, edges: { width: 1, smooth: false },
  });
  network.on("click", (p) => {
    const id = p.nodes[0];
    if (typeof id === "number") { $("#r-game").value = id; $("#r-user").value = ""; runReco(); }
  });
}

async function runReco(user) {
  const id = $("#r-game").value;
  if (!id) return;
  const liked = $("#r-liked").value;
  const r = await api(`/api/recommend?app_id=${id}&liked=${liked}`);
  $("#r-userbox").innerHTML = user
    ? `<div class="userbox"><b>${user.user_id}</b> (${esc(user.reviewer_tier)}) đã review <b>${esc(user.game_name)}</b> ${user.voted_up ? "👍" : "👎"}<br>
       <i>“${esc(user.review_text)}”</i></div>` : "";
  $("#r-list").innerHTML = `<h3>${r.liked ? "Thích" : "Không thích"} ${esc(r.source.game_name)} → gợi ý</h3>
    <p class="note">${esc(r.source.genre)} · ${esc(r.source.publisher || "chưa rõ NPH")}. Điểm = 2 × cùng thể loại + 1 × cùng NPH, hoà thì xếp theo % tích cực.</p>`
    + (r.items.length ? r.items.map((x, i) => `<div class="reco"><div class="rank">${i + 1}</div>
        <div><div class="name">${esc(x.game_name)}</div><div class="why">${x.why.map(esc).join(" · ")}</div></div>
        <div class="score">${x.game_pos_pct}% 👍</div></div>`).join("") : '<div class="empty">Không có game cùng nhóm.</div>');
  const hit = new Set(r.items.map((x) => x.app_id));
  nodes.update(groups.games.map((g) => ({ id: g.app_id, borderWidth: g.app_id == id ? 5 : hit.has(g.app_id) ? 3 : 1,
    color: { background: genreColor[g.genre], border: g.app_id == id ? "#000" : hit.has(g.app_id) ? "#c0392b" : genreColor[g.genre] },
    opacity: g.app_id == id || hit.has(g.app_id) ? 1 : .35 })));
  network.selectNodes([+id]); network.focus(+id, { scale: .8, animation: true });
}
