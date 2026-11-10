/* Front end: talks to the Flask REST API using fetch(). */
const $ = (id) => document.getElementById(id);
const rs = (n) => (n === null || n === undefined ? "-" : "Rs. " + Math.round(n).toLocaleString("en-IN"));
const page = document.body.dataset.page;
async function api(url, opts = {}) {
  if (opts.json !== undefined) { opts.method = opts.method || "POST"; opts.headers = { "Content-Type": "application/json" }; opts.body = JSON.stringify(opts.json); }
  const r = await fetch(url, opts); let data = {};
  try { data = await r.json(); } catch (e) {}
  if (!r.ok) { if (r.status === 401 && page !== "login") location.href = "/login"; throw new Error(data.error || "Request failed"); }
  return data;
}
const say = (id, text, cls) => { const e = $(id); if (e) { e.textContent = text; e.className = "msg " + (cls || ""); } };
const tag = (s) => `<span class="tag ${s}">${s}</span>`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const monthNow = () => new Date().toISOString().slice(0, 7);
let charts = {};
function chart(id, cfg) { if (charts[id]) charts[id].destroy(); if (window.Chart) charts[id] = new Chart($(id), cfg); }

const pages = {
  login() {
    const go = async () => { try { await api("/api/login", { json: { email: $("email").value, password: $("password").value } }); location.href = "/dashboard"; } catch (e) { say("msg", e.message, "err"); } };
    $("loginBtn").onclick = go; $("password").onkeydown = (e) => e.key === "Enter" && go();
  },
  register() {
    $("regBtn").onclick = async () => { try { await api("/api/register", { json: { name: $("name").value, email: $("email").value, password: $("password").value, monthly_income: $("income").value } }); location.href = "/dashboard"; } catch (e) { say("msg", e.message, "err"); } };
  },
  async dashboard_page() {
    const s = await api("/api/summary?month=" + monthNow());
    $("cSpent").textContent = rs(s.total); $("cIncome").textContent = rs(s.income); $("cAlerts").textContent = s.unread_alerts; $("monthLbl").textContent = s.month; $("incomeInput").value = s.income;
    $("catRows").innerHTML = s.by_category.map((r) => `<tr><td>${r.category}</td><td>${r.type}</td><td>${rs(r.spent)}</td><td>${rs(r.budget)}</td>
      <td><div class="bar"><i style="width:${Math.min(r.percent || 0, 100)}%;background:${r.status === "EXCEEDED" ? "#d9534f" : r.status === "WARNING" ? "#e07b39" : r.status === "CAUTION" ? "#e0a800" : "#2e8b57"}"></i></div></td><td>${tag(r.status === "NO BUDGET" ? "-" : r.status)}</td></tr>`).join("");
    chart("barChart", { type: "bar", data: { labels: s.by_category.map((r) => r.category), datasets: [{ label: "Spent (Rs.)", data: s.by_category.map((r) => r.spent), backgroundColor: "#6c8ebf" }] }, options: { plugins: { legend: { display: false } } } });
    const f = await api("/api/forecast?months=6");
    if (f.available) {
      $("cPred").textContent = rs(f.total[0]);
      chart("lineChart", { type: "line", data: { labels: [...f.history.months, ...f.months], datasets: [{ label: "History", data: [...f.history.total, ...f.months.map(() => null)], borderColor: "#222", tension: .2 },
        { label: "Forecast", data: [...f.history.months.map((_, i) => (i === f.history.months.length - 1 ? f.history.total[i] : null)), ...f.total], borderColor: "#d9534f", borderDash: [6, 4], tension: .2 }] } });
    } else { $("cPred").textContent = "Need data"; say("fcMsg", f.message, "muted"); }
    $("saveIncome").onclick = async () => { try { await api("/api/me", { method: "PUT", json: { monthly_income: $("incomeInput").value } }); say("setMsg", "Income saved", "okmsg"); $("cIncome").textContent = rs($("incomeInput").value); } catch (e) { say("setMsg", e.message, "err"); } };
    $("delAcc").onclick = async () => { if (!confirm("This permanently deletes your account and all data. Continue?")) return; try { await api("/api/account", { method: "DELETE", json: { password: $("delPw").value } }); location.href = "/login"; } catch (e) { say("setMsg", e.message, "err"); } };
  },
  add_page() {
    $("date").value = new Date().toISOString().slice(0, 10); let t;
    $("description").oninput = () => { clearTimeout(t); t = setTimeout(async () => { const d = $("description").value.trim(); if (!d) { $("predBox").textContent = "type a description..."; return; }
      try { const p = await api("/api/predict?description=" + encodeURIComponent(d)); $("predBox").textContent = `${p.category} (${Math.round(p.confidence * 100)}% confidence)`; } catch (e) {} }, 250); };
    $("saveBtn").onclick = async () => {
      try {
        const r = await api("/api/expenses", { json: { amount: $("amount").value, date: $("date").value, description: $("description").value, time: $("time").value, payment_mode: $("mode").value, category: $("category").value } });
        say("msg", "Saved.", "okmsg");
        $("result").innerHTML = `<p>Category: <b>${r.category}</b> (confidence ${Math.round(r.confidence * 100)}%)</p><p>Anomaly check: ${r.anomaly ? '<span class="flag">UNUSUAL - please confirm this transaction</span>' : "Normal"}</p><p>Budget status: ${tag(r.budget_status === "NO BUDGET" ? "-" : r.budget_status)}</p>` +
          r.alerts.map((a) => `<div class="alert ${a.level}"><b>${a.level}</b> ${esc(a.message)}</div>`).join("");
        $("description").value = ""; $("amount").value = ""; $("predBox").textContent = "type a description...";
      } catch (e) { say("msg", e.message, "err"); }
    };
  },
  transactions_page() {
    $("month").value = monthNow();
    const load = async () => {
      const rows = await api("/api/expenses?month=" + $("month").value);
      $("rows").innerHTML = rows.map((r) => `<tr><td>${r.date}</td><td>${esc(r.description)}</td><td><select data-id="${r.id}" class="catSel">${["Uncategorised", ...window.CATEGORIES].map((c) => `<option ${c === r.category ? "selected" : ""}>${c}</option>`).join("")}</select></td>
        <td>${rs(r.amount)}</td><td>${r.anomaly ? '<span class="flag">FLAGGED</span>' : "Normal"}</td><td><button class="red small" data-del="${r.id}">Delete</button></td></tr>`).join("") || '<tr><td colspan="6">No transactions in this month</td></tr>';
      document.querySelectorAll(".catSel").forEach((s) => (s.onchange = async () => { if (s.value === "Uncategorised") return; try { await api("/api/expenses/" + s.dataset.id, { method: "PUT", json: { category: s.value } }); say("msg", "Category updated and stored for re-training.", "okmsg"); } catch (e) { say("msg", e.message, "err"); } }));
      document.querySelectorAll("[data-del]").forEach((b) => (b.onclick = async () => { await api("/api/expenses/" + b.dataset.del, { method: "DELETE" }); load(); }));
    };
    $("month").onchange = load; load();
    $("uploadBtn").onclick = async () => { const f = $("file").files[0]; if (!f) return say("msg", "Choose a CSV file first", "err"); const fd = new FormData(); fd.append("file", f);
      try { const r = await api("/api/upload", { method: "POST", body: fd }); say("msg", `Imported ${r.imported} rows, skipped ${r.skipped}. ` + r.errors.map((e) => `Row ${e.row}: ${e.error}`).join("; "), r.skipped ? "err" : "okmsg"); load(); } catch (e) { say("msg", e.message, "err"); } };
    $("retrainBtn").onclick = async () => { try { const r = await api("/api/retrain", { method: "POST" }); say("msg", `Personal categoriser re-trained with ${r.retrained_with} correction(s).`, "okmsg"); } catch (e) { say("msg", e.message, "err"); } };
  },
  budget_page() {
    let last = null;
    const draw = (cats) => { $("rows").innerHTML = cats.map((c) => `<tr><td>${c.category}</td><td>${c.type}</td><td>${rs(c.forecast)}</td><td><input class="bud" data-c="${c.category}" type="number" value="${c.budget ?? ""}" style="max-width:140px"></td></tr>`).join(""); };
    (async () => { const me = await api("/api/me"); $("income").value = me.monthly_income;
      const b = await api("/api/budget?month=" + monthNow()); const names = Object.keys(b.budgets);
      if (names.length) { draw((await api("/api/categories")).map((c) => ({ category: c.name, type: c.type, forecast: null, budget: b.budgets[c.name] ?? 0 }))); $("info").textContent = "Saved budget for " + b.month; } })();
    const rec = async (save) => { try { const r = await api("/api/budget/recommend", { json: { income: $("income").value, savings_rate: $("rate").value / 100, save } }); last = r; draw(r.categories);
      $("info").textContent = `Recommended budget for ${r.month} (${r.method})`; $("tot").textContent = `Spendable: ${rs(r.spendable)} | Total forecast: ${rs(r.total_forecast)} | Recommended total: ${rs(r.total_budget)}`; say("msg", save ? "Budget saved." : "Preview only - press Save to keep it.", "okmsg"); } catch (e) { say("msg", e.message, "err"); } };
    $("recBtn").onclick = () => rec(false); $("saveRec").onclick = () => rec(true);
    $("saveManual").onclick = async () => { const budgets = {}; document.querySelectorAll(".bud").forEach((i) => (budgets[i.dataset.c] = i.value || 0));
      try { await api("/api/budget", { method: "PUT", json: { month: monthNow(), budgets } }); say("msg", "Budgets saved.", "okmsg"); } catch (e) { say("msg", e.message, "err"); } };
  },
  forecast_page() {
    const load = async () => { const f = await api("/api/forecast?months=" + $("n").value);
      if (!f.available) { say("msg", f.message, "err"); return; } say("msg", "Model: " + f.model + ". Estimates only - uncertainty grows with every month.", "muted");
      chart("chart", { type: "line", data: { labels: [...f.history.months, ...f.months], datasets: [{ label: "History", data: [...f.history.total, ...f.months.map(() => null)], borderColor: "#222" },
        { label: "Forecast", data: [...f.history.months.map((_, i) => (i === f.history.months.length - 1 ? f.history.total[i] : null)), ...f.total], borderColor: "#d9534f", borderDash: [6, 4] }] } });
      $("head").innerHTML = "<tr><th>Category</th>" + f.months.map((m) => `<th>${m}</th>`).join("") + "</tr>";
      $("rows").innerHTML = Object.entries(f.categories).map(([c, v]) => `<tr><td>${c}</td>${v.map((x) => `<td>${rs(x)}</td>`).join("")}</tr>`).join("") + `<tr><td><b>Total</b></td>${f.total.map((x) => `<td><b>${rs(x)}</b></td>`).join("")}</tr>`; };
    $("n").onchange = load; load();
  },
  alerts_page() {
    const load = async () => { const a = await api("/api/alerts");
      $("list").innerHTML = a.map((x) => `<div class="alert ${x.level} ${x.read ? "read" : ""}"><b style="min-width:80px">${x.level}</b><span style="flex:1">${esc(x.message)} <span class="muted">(${x.created_at})</span></span>${x.read ? "" : `<button class="small grey" data-r="${x.id}">Mark as reviewed</button>`}</div>`).join("") || "<p>No alerts yet.</p>";
      document.querySelectorAll("[data-r]").forEach((b) => (b.onclick = async () => { await api("/api/alerts/" + b.dataset.r + "/review", { method: "PUT" }); load(); })); };
    load();
  },
};
pages.dashboard_page && (pages[page] || (() => {}))();
