// Corpus operations use structured API requests, never command strings.
let corpusData = { corpora: [], jobs: [] };
let corpusOffset = 0;
let corpusViewedJob = null;
let corpusLoadVersion = 0;
const corpusActionNames = { prepare: "Chuẩn bị dữ liệu", verify: "Kiểm tra dữ liệu", ingest: "Extract + embed", benchmark: "Đo tìm kiếm từ khóa" };
const corpusStatusNames = { queued: "Đang chờ", running: "Đang chạy", stopping: "Đang dừng", completed: "Đã hoàn tất", completed_with_errors: "Hoàn tất có lỗi", stopped: "Đã dừng", interrupted: "Bị gián đoạn", failed: "Thất bại" };
const corpusNumber = value => Number(value || 0).toLocaleString("vi-VN");
const corpusActive = job => ["queued", "running", "stopping"].includes(job.status);

function corpusError(message = "") {
  $("#corpus-error").textContent = message;
  $("#corpus-error").hidden = !message;
}

async function loadCorpus(loadPreview = true) {
  const version = ++corpusLoadVersion;
  clearTimeout(corpusPoll);
  try {
    const data = await api("/corpus");
    if (version !== corpusLoadVersion) return;
    const hadActive = corpusData.jobs.some(corpusActive);
    corpusData = data;
    const previous = $("#corpus-select").value;
    $("#corpus-select").innerHTML = data.corpora.map(c => `<option value="${esc(c.id)}">${esc(c.id)} · ${corpusNumber(c.records)} hồ sơ</option>`).join("");
    $("#corpus-select").value = data.corpora.some(c => c.id === previous) ? previous : data.corpora[0]?.id || "";
    renderCorpusSummary();
    renderCorpusHistory();
    const active = data.jobs.find(corpusActive);
    if (active) corpusViewedJob = active.id;
    if (!corpusViewedJob) corpusViewedJob = data.jobs[0]?.id;
    const job = data.jobs.find(j => j.id === corpusViewedJob);
    if (job) renderCorpusJob(job);
    else $("#corpus-job").hidden = true;
    if (loadPreview || (hadActive && !active)) await loadCorpusPreview();
    if (corpusVisible && active) corpusPoll = setTimeout(() => loadCorpus(false), 1500);
  } catch (error) {
    corpusError(error.message);
    if (corpusVisible) corpusPoll = setTimeout(() => loadCorpus(false), 5000);
  }
}

function renderCorpusSummary() {
  const selected = corpusData.corpora.find(c => c.id === $("#corpus-select").value);
  const counts = corpusData.application_counts || {};
  $("#corpus-summary").innerHTML = [
    ["Hồ sơ trong tập đang chọn", selected?.records || 0],
    ["Vị trí tiếp tục", selected ? (selected.next_record >= selected.records ? "Đã hết" : selected.next_record + 1) : 0],
    ["CV sẵn sàng trong ứng dụng", counts.ready || 0], ["CV lỗi trong ứng dụng", counts.failed || 0],
  ].map(([label, value]) => `<div class="stat-card"><span class="muted">${label}</span><strong>${typeof value === "number" ? corpusNumber(value) : esc(value)}</strong></div>`).join("");
  $("#corpus-source").textContent = corpusData.source_available ? "Đã có dữ liệu nguồn trên máy" : "Chưa có dữ liệu nguồn để tạo tập mới";
  const busy = corpusData.jobs.some(corpusActive);
  document.querySelectorAll("[data-corpus-action]").forEach(button => {
    button.disabled = busy || (button.dataset.corpusAction === "prepare" ? !corpusData.source_available : !selected);
  });
}

async function loadCorpusPreview() {
  const id = $("#corpus-select").value;
  if (!id) {
    $("#corpus-preview").innerHTML = `<tr><td colspan="5" class="empty">Chưa có tập hồ sơ. Tạo tập mới từ dữ liệu nguồn để bắt đầu.</td></tr>`;
    $("#corpus-page").textContent = "";
    $("#corpus-prev").disabled = $("#corpus-next").disabled = true;
    return;
  }
  const offset = corpusOffset;
  const page = await api(`/corpus/${encodeURIComponent(id)}/preview?offset=${offset}&limit=10`);
  if (id !== $("#corpus-select").value || offset !== corpusOffset) return;
  const statuses = { ready: "Sẵn sàng", failed: "Lỗi", not_processed: "Chưa xử lý", invalid: "Dữ liệu lỗi" };
  $("#corpus-preview").innerHTML = page.rows.map(row => `<tr><td>${row.record}</td><td>${esc(row.name)}</td><td>${esc(row.position || "—")}</td><td><span class="badge s-${esc(row.status)}">${esc(statuses[row.status])}</span></td><td class="corpus-excerpt">${esc(row.excerpt)}</td></tr>`).join("") || `<tr><td colspan="5" class="empty">Không có hồ sơ ở trang này.</td></tr>`;
  $("#corpus-page").textContent = page.rows.length ? `Hồ sơ ${corpusNumber(offset + 1)}–${corpusNumber(offset + page.rows.length)} / ${corpusNumber(page.total)}` : "Đã hết tập hồ sơ";
  $("#corpus-prev").disabled = offset === 0;
  $("#corpus-next").disabled = offset + page.rows.length >= page.total;
}

function renderCorpusHistory() {
  $("#corpus-history").innerHTML = corpusData.jobs.map(job => `<tr>
    <td>${esc(corpusActionNames[job.action])}<div class="muted">${esc(new Date(job.created_at).toLocaleString("vi-VN"))}</div></td>
    <td>${esc(job.params.corpus_id || job.result?.corpus_id || "Tập mới")}</td>
    <td>${esc(corpusStatusNames[job.status])}</td>
    <td>${job.action === "ingest" ? `${job.ready} sẵn sàng · ${job.failed} lỗi · ${job.skipped} bỏ qua` : esc(job.result?.records ? `${corpusNumber(job.result.records)} hồ sơ` : job.phase)}</td>
    <td><button class="link" data-corpus-job="${job.id}">Xem kết quả</button></td></tr>`).join("") || `<tr><td colspan="5" class="empty">Chưa chạy thao tác corpus từ giao diện.</td></tr>`;
}

function renderCorpusJob(job) {
  const element = $("#corpus-job");
  element.hidden = false;
  const active = corpusActive(job);
  const progress = Number(job.progress || 0), total = Number(job.total || 0);
  const percent = total > 0 ? Math.min(100, Math.floor(progress * 100 / total)) : null;
  const result = job.result || {};
  element.innerHTML = `<div class="row-between"><h3>${esc(corpusActionNames[job.action])} · ${esc(corpusStatusNames[job.status])}</h3>
    <a href="/corpus/jobs/${job.id}/report" download>Tải báo cáo JSON</a></div>
    <p>${esc(job.phase)}</p>
    <progress aria-label="Tiến độ tác vụ corpus" max="100" ${percent === null ? "" : `value="${percent}"`}></progress>
    <p class="muted">${total ? `${corpusNumber(progress)} / ${corpusNumber(total)} ${esc(job.unit || "hồ sơ")}` : `${corpusNumber(progress)} ${esc(job.unit || "hồ sơ")}`} · ${corpusNumber(job.elapsed_seconds)} giây${active ? " · Tự cập nhật" : ""}</p>
    ${job.action === "ingest" ? `<div class="corpus-result-counts"><span>Đã xử lý: <b>${job.processed}</b></span><span>Sẵn sàng: <b>${job.ready}</b></span><span>Lỗi: <b>${job.failed}</b></span><span>Đã có, bỏ qua: <b>${job.skipped}</b></span></div>
    <p class="muted">Đã duyệt ${job.scanned} hồ sơ nguồn. Tiến độ tăng sau khi lưu xong mỗi lô. Dừng sẽ chờ lô đang chạy hoàn tất.</p>` : ""}
    ${result.checksum_matches ? `<p class="s-ready">Dữ liệu khớp bản chuẩn bị · ${corpusNumber(result.records)} hồ sơ.</p>` : ""}
    ${result.corpus_id ? `<p>Đã tạo tập <b>${esc(result.corpus_id)}</b> với ${corpusNumber(result.records)} hồ sơ.</p>` : ""}
    ${result.end_of_corpus ? `<p>Đã đến cuối tập hồ sơ. Hồ sơ lỗi có thể thử lại bằng chế độ chọn vị trí.</p>` : ""}
    ${result.queries ? `<h3>Kết quả benchmark</h3><p>${corpusNumber(result.records)} hồ sơ · ${result.query_repeats} lần/truy vấn · dựng chỉ mục ${result.build_seconds} giây · bộ nhớ đệm đã ấm.</p>
    <div class="table-wrap"><table><thead><tr><th>Truy vấn</th><th>Hồ sơ khớp</th><th>Trung vị (ms)</th><th>p95 (ms)</th></tr></thead><tbody>${result.queries.map(q => `<tr><td>${esc(q.query)}</td><td>${corpusNumber(q.matches)}</td><td>${q.p50_ms}</td><td>${q.p95_ms}</td></tr>`).join("")}</tbody></table></div>
    <p class="muted">Tìm kiếm từ khóa cục bộ. Chưa đánh giá độ phù hợp CV–JD và không gồm độ trễ LLM, embedding hoặc HTTP.</p>` : ""}
    ${job.errors?.length ? `<details><summary>${job.errors.length} lỗi gần nhất</summary><ul>${job.errors.map(e => `<li>${esc(e.cv_id || `Dòng ${e.record}`)}: ${esc(e.message)}</li>`).join("")}</ul></details>` : ""}
    ${active ? `<button id="corpus-stop" type="button" ${job.status === "stopping" ? "disabled" : ""}>Dừng sau lô hiện tại</button>` : ""}`;
  if (active) $("#corpus-stop").addEventListener("click", async () => {
    try { await api(`/corpus/jobs/${job.id}/stop`, { method: "POST" }); await loadCorpus(false); }
    catch (error) { corpusError(error.message); }
  });
}

async function submitCorpus(action) {
  corpusError();
  const payload = { action, corpus_id: $("#corpus-select").value || null };
  if (action === "prepare") Object.assign(payload, { sample_count: Number($("#corpus-sample-count").value), seed: Number($("#corpus-seed").value) });
  if (action === "ingest") Object.assign(payload, { limit: Number($("#corpus-limit").value), batch_size: Number($("#corpus-batch-size").value), mode: $("#corpus-mode").value, offset: Number($("#corpus-offset").value) - 1 });
  if (action === "benchmark") payload.repeats = Number($("#corpus-repeats").value);
  document.querySelectorAll("[data-corpus-action]").forEach(button => { button.disabled = true; });
  try {
    const job = await api("/corpus/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    corpusViewedJob = job.id;
    await loadCorpus(false);
  } catch (error) { corpusError(error.message); renderCorpusSummary(); }
}

for (const action of Object.keys(corpusActionNames)) $("#corpus-" + action + "-form").addEventListener("submit", event => { event.preventDefault(); submitCorpus(action); });
$("#corpus-mode").addEventListener("change", () => { $("#corpus-offset-label").hidden = $("#corpus-mode").value !== "range"; });
$("#corpus-select").addEventListener("change", () => { corpusOffset = 0; renderCorpusSummary(); loadCorpusPreview().catch(error => corpusError(error.message)); });
$("#corpus-refresh").addEventListener("click", () => loadCorpus());
$("#corpus-prev").addEventListener("click", () => { corpusOffset = Math.max(0, corpusOffset - 10); loadCorpusPreview().catch(error => corpusError(error.message)); });
$("#corpus-next").addEventListener("click", () => { corpusOffset += 10; loadCorpusPreview().catch(error => corpusError(error.message)); });
$("#corpus-history").addEventListener("click", event => { const id = event.target.closest("[data-corpus-job]")?.dataset.corpusJob; if (id) { corpusViewedJob = id; renderCorpusJob(corpusData.jobs.find(j => j.id === id)); } });
$("#corpus-search").addEventListener("click", () => showTab("search"));
if (location.hash === "#corpus") showTab("corpus");
