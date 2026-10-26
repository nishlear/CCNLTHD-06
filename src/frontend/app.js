const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const badge = (s) => `<span class="badge s-${esc(s)}">${esc(s)}</span>`;
const chips = (arr, cls = "") => (arr || []).map((x) => `<span class="chip ${cls}">${esc(x)}</span>`).join("");

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    throw new Error(typeof d === "string" ? d : JSON.stringify(d) || res.statusText);
  }
  return data;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), 3500);
}

// ---------- tabs ----------
function showTab(name) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  if (name === "cvs") loadCVs();
  if (name === "history") loadHistory();
}
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ---------- CVs ----------
const CV_PAGE_SIZE = 100;
let cvOffset = 0;
let cvLoadVersion = 0;
const uploadInput = $("#upload-input");
const drop = uploadInput.parentElement;
const updateUploadLabel = () => {
  const n = uploadInput.files.length;
  $("#upload-label").textContent = n ? `${n} file đã chọn` : "Chọn hoặc kéo thả các file CV (PDF)";
};
uploadInput.addEventListener("change", updateUploadLabel);
["dragover", "dragenter"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => drop.addEventListener(e, () => drop.classList.remove("over")));
drop.addEventListener("drop", (ev) => {
  ev.preventDefault();
  uploadInput.files = ev.dataTransfer.files;
  updateUploadLabel();
});

$("#upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!uploadInput.files.length) return toast("Chưa chọn file nào");
  const fd = new FormData();
  for (const f of uploadInput.files) fd.append("files", f);
  const btn = e.submitter;
  btn.disabled = true;
  btn.textContent = "Đang xử lý…";
  try {
    const { cvs } = await api("/cvs", { method: "POST", body: fd });
    const ok = cvs.filter((c) => c.status === "ready").length;
    toast(`${ok}/${cvs.length} CV sẵn sàng`);
    uploadInput.value = "";
    updateUploadLabel();
    cvOffset = 0;
    loadCVs();
  } catch (err) {
    toast(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Tải lên";
  }
});

$("#cv-filter").addEventListener("change", () => { cvOffset = 0; loadCVs(); });
$("#cv-prev").addEventListener("click", () => { cvOffset = Math.max(0, cvOffset - CV_PAGE_SIZE); loadCVs(); });
$("#cv-next").addEventListener("click", () => { cvOffset += CV_PAGE_SIZE; loadCVs(); });

async function loadCVs() {
  try {
    const status = $("#cv-filter").value;
    const version = ++cvLoadVersion;
    const page = await api(`/cvs?limit=${CV_PAGE_SIZE + 1}&offset=${cvOffset}${status ? `&status=${status}` : ""}`);
    if (version !== cvLoadVersion) return;
    const cvs = page.slice(0, CV_PAGE_SIZE);
    $("#cv-prev").disabled = cvOffset === 0;
    $("#cv-next").disabled = page.length <= CV_PAGE_SIZE;
    $("#cv-page").textContent = cvs.length ? `CV ${cvOffset + 1}–${cvOffset + cvs.length}` : "Không có CV ở trang này";
    $("#cv-rows").innerHTML = cvs.length
      ? cvs.map((c) => `
        <tr>
          <td><button class="link" data-view="${esc(c.id)}">${esc(c.full_name || c.file_name)}</button></td>
          <td>${esc(c.file_name)}</td>
          <td>${chips(c.skills)}</td>
          <td>${c.years_experience ?? "—"}</td>
          <td>${badge(c.status)}${c.error ? `<div class="err">${esc(c.error)}</div>` : ""}</td>
          <td><button class="link danger" data-del="${c.id}">Xóa</button></td>
        </tr>`).join("")
      : `<tr><td colspan="6" class="empty">Chưa có CV nào</td></tr>`;
  } catch (err) { toast(err.message); }
}

$("#cv-rows").addEventListener("click", async (e) => {
  const viewId = e.target.closest("[data-view]")?.dataset.view;
  if (viewId) return openCV(viewId);
  const id = e.target.dataset.del;
  if (!id || !confirm("Xóa CV này?")) return;
  try {
    await api(`/cvs/${id}`, { method: "DELETE" });
    loadCVs();
  } catch (err) {
    toast(err.message);
  }
});

// ---------- Search ----------
$("#search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = e.submitter;
  btn.disabled = true;
  $("#search-view").innerHTML = `<p class="muted">Đang tìm…</p>`;
  try {
    const res = await api("/searches", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: $("#search-query").value, top_k: Number($("#search-topk").value) }),
    });
    await openSearch(res.id);
  } catch (err) {
    $("#search-view").innerHTML = "";
    toast(err.message);
  } finally {
    btn.disabled = false;
  }
});

async function openSearch(id) {
  showTab("search");
  const s = await api(`/searches/${id}`);
  $("#search-query").value = s.query;
  $("#search-topk").value = s.top_k;
  renderSearch(s);
  history.replaceState(null, "", `#search=${encodeURIComponent(id)}`);
}

function searchBadge(s) {
  const labels = { succeeded: "Hoàn tất thành công", completed_with_errors: "Hoàn tất có lỗi", failed: "Thất bại" };
  return s.outcome && labels[s.outcome]
    ? `<span class="badge s-${esc(s.outcome)}">${esc(labels[s.outcome])}</span>` : badge(s.status);
}

function renderSearch(s) {
  const reviewing = s.status === "awaiting_review";
  const c = s.criteria || {};
  const crit = s.criteria
    ? `<span>Kỹ năng: ${chips(c.skills) || "—"}</span>
       <span>Tối thiểu: ${c.min_years ?? "—"} năm</span>
       <span>Cấp bậc: ${esc(c.level ?? "—")}</span>
       <span>Từ khóa: ${chips(c.keywords) || "—"}</span>`
    : "";

  const rows = s.shortlist.map((r) => `
    <tr>
      <td>${reviewing
        ? `<input type="checkbox" class="approve" value="${esc(r.cv_id)}" ${r.score >= 0.5 ? "checked" : ""}>`
        : r.approved ? "✓" : ""}</td>
      <td>${r.rank}</td>
      <td>${esc(r.full_name || r.cv_id)}</td>
      <td><span class="score">${r.score.toFixed(2)}</span><div class="bar"><i style="width:${r.score * 100}%"></i></div></td>
      <td>${chips(r.matched_skills)}${chips(r.missing_skills, "miss")}</td>
      <td>${r.reasons ? esc(r.reasons) : '<span class="muted">—</span>'}</td>
    </tr>`).join("");

  $("#search-view").innerHTML = `
    <div class="meta">${searchBadge(s)} ${crit}</div>
    <p class="muted">Điểm dùng để xếp hạng ưu tiên; điểm cao chưa bảo đảm đạt mọi yêu cầu. Kinh nghiệm tối thiểu là điểm mềm. Ô chọn sẵn từ 0,5 cần được HR kiểm tra lại.</p>
    ${s.shortlist.length ? `
      <table>
        <thead><tr><th>${reviewing ? "Giữ" : "Duyệt"}</th><th>#</th><th>Ứng viên</th><th>Điểm</th><th>Kỹ năng</th><th>Lý do</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>` : `<p class="empty">Không có ứng viên phù hợp</p>`}
    ${reviewing ? `
      <div class="actions">
        <button class="primary" id="submit-review">Gửi duyệt &amp; tạo câu hỏi</button>
        <span class="muted">Workflow đang tạm dừng chờ bạn chọn.</span>
      </div>` : ""}
    <div id="questions">${renderQuestions(s.questions, s.shortlist)}${renderQuestionStatus(s)}</div>`;

  if (reviewing) $("#submit-review").addEventListener("click", () => submitReview(s.id));
}

function renderQuestions(questions, shortlist) {
  if (!questions?.length) return "";
  const names = Object.fromEntries(shortlist.map((r) => [r.cv_id, r.full_name || r.cv_id]));
  const byCv = {};
  for (const q of questions) (byCv[q.cv_id] ||= []).push(q);
  return `<h2>Câu hỏi phỏng vấn</h2>` + Object.entries(byCv).map(([cv, qs]) => `
    <div class="q-group">
      <h3>${esc(names[cv] || cv)}</h3>
      <ul>${qs.map((q) => `<li><span class="cat">${esc(q.category)}</span> — ${esc(q.question)}</li>`).join("")}</ul>
    </div>`).join("");
}

async function submitReview(id) {
  const ids = [...document.querySelectorAll(".approve:checked")].map((x) => x.value);
  const btn = $("#submit-review");
  btn.disabled = true;
  btn.textContent = "Đang tạo câu hỏi…";
  try {
    await api(`/searches/${id}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approved_cv_ids: ids }),
    });
    await openSearch(id);
  } catch (err) {
    toast(err.message);
    btn.disabled = false;
    btn.textContent = "Gửi duyệt & tạo câu hỏi";
  }
}

// ---------- History ----------
async function loadHistory() {
  try {
    const list = await api("/searches");
    const counts = list.reduce((out, row) => { out[row.outcome || "pending"] = (out[row.outcome || "pending"] || 0) + 1; return out; }, {});
    $("#history-summary").textContent = `Thành công: ${counts.succeeded || 0} · Hoàn tất có lỗi: ${counts.completed_with_errors || 0} · Thất bại: ${counts.failed || 0} · Đang xử lý/chờ duyệt: ${counts.pending || 0}`;
    $("#history-rows").innerHTML = list.length
      ? list.map((s) => `
        <tr class="clickable" data-id="${s.id}">
          <td>${esc(s.query)}</td>
          <td>${s.top_k}</td>
          <td>${searchBadge(s)}</td>
          <td class="muted">${new Date(s.created_at).toLocaleString()}</td>
        </tr>`).join("")
      : `<tr><td colspan="4" class="empty">Chưa có lượt tìm kiếm nào</td></tr>`;
  } catch (err) { toast(err.message); }
}

$("#history-rows").addEventListener("click", (e) => {
  const tr = e.target.closest("tr[data-id]");
  if (tr) openSearch(tr.dataset.id).catch((err) => toast(err.message));
});

function renderQuestionStatus(search) {
  if (search.status !== "completed") return "";
  const generated = new Set((search.questions || []).map((q) => q.cv_id));
  const missing = search.shortlist.filter((r) => r.approved && !generated.has(r.cv_id));
  const missingMessage = missing.length
    ? `<p class="err">Chưa tạo được câu hỏi cho: ${missing.map((r) => esc(r.full_name || r.cv_id)).join(", ")}.</p>` : "";
  const warning = s => s.outcome === "completed_with_errors" || (s.errors || []).length;
  const details = (search.errors || []).map(e => `${esc(e.node)}: ${esc(e.code)}`).join("; ");
  return missingMessage + (warning(search)
    ? `<p class="err">Hoàn tất có lỗi. Câu hỏi: ${esc(search.question_count ?? (search.questions || []).length)}/${esc(search.expected_question_count ?? "—")}. ${details}</p>` : "");
}

async function openCV(id) {
  try {
    const cv = await api(`/cvs/${encodeURIComponent(id)}`);
    $("#cv-detail").innerHTML = `
      <h3>${esc(cv.full_name || cv.file_name)}</h3>
      <p>${badge(cv.status)} ${esc(cv.file_name)}</p>
      <p>${chips(cv.skills)}</p>
      <dl>${[
        ["Email", cv.email], ["Điện thoại", cv.phone],
        ["Năm kinh nghiệm", cv.years_experience], ["Học vấn", cv.education],
        ["Tóm tắt", cv.summary], ["Lỗi xử lý", cv.error],
      ].map(([label, value]) => `<dt>${esc(label)}</dt><dd>${esc(value ?? "Chưa có dữ liệu")}</dd>`).join("")}</dl>`;
    $("#cv-dialog").showModal();
  } catch (err) { toast(err.message); }
}

$("#close-cv").addEventListener("click", () => $("#cv-dialog").close());
const initialSearch = location.hash.match(/^#search=(.+)$/);
if (initialSearch) {
  openSearch(decodeURIComponent(initialSearch[1])).catch((err) => toast(err.message));
} else {
  loadCVs();
}
