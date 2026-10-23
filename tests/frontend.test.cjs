const test = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");
const fs = require("node:fs");
const path = require("node:path");

function frontend(responses = {}) {
  const elements = new Map();
  const element = (selector) => {
    if (!elements.has(selector)) elements.set(selector, {
      value: "", innerHTML: "", textContent: "", files: [],
      addEventListener() {}, classList: { toggle() {}, add() {}, remove() {} },
      showModal() { this.open = true; }, close() { this.open = false; },
      parentElement: { addEventListener() {}, classList: { add() {}, remove() {} } },
    });
    return elements.get(selector);
  };
  const history = { replaceState(_state, _title, url) { this.url = url; } };
  const context = vm.createContext({
    document: { querySelector: element, querySelectorAll: () => [] },
    location: { hash: "" }, history, setTimeout, clearTimeout,
    fetch: async (url) => ({ status: 200, ok: true, json: async () => responses[url] || responses[url.split("?")[0]] || [] }),
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../src/frontend/app.js"), "utf8"), context);
  return { context, element, history };
}

test("CV list opens a detail action and escapes names", async () => {
  const { context, element } = frontend({ "/cvs": [{ id: "cv-a", full_name: "<Candidate>",
    file_name: "cv.pdf", status: "ready", skills: ["Python"], years_experience: null }] });
  await vm.runInContext("loadCVs()", context);
  assert.match(element("#cv-rows").innerHTML, /data-view="cv-a"/);
  assert.match(element("#cv-rows").innerHTML, /&lt;Candidate&gt;/);
});

test("CV detail displays unknown data and opens its dialog", async () => {
  const { context, element } = frontend({ "/cvs/cv-a": { id: "cv-a", full_name: "Candidate A",
    file_name: "cv.pdf", status: "ready", skills: ["Python"], years_experience: null } });
  await vm.runInContext("openCV('cv-a')", context);
  assert.equal(element("#cv-dialog").open, true);
  assert.match(element("#cv-detail").innerHTML, /Chưa có dữ liệu/);
});

test("saved search has a reopen URL and identifies missing question output", async () => {
  const { context, element, history } = frontend({ "/searches/s1": {
    id: "s1", query: "Python", top_k: 1, status: "completed", criteria: {}, questions: [],
    shortlist: [{ cv_id: "a", full_name: "Candidate A", score: .8, rank: 1,
      approved: true, matched_skills: [], missing_skills: [], reasons: null }],
  } });
  await vm.runInContext("openSearch('s1')", context);
  assert.equal(history.url, "#search=s1");
  assert.match(element("#search-view").innerHTML, /Chưa tạo được câu hỏi cho: Candidate A/);
});

test("CV pagination renders at most one page and enables next only when more rows exist", async () => {
  const rows = Array.from({ length: 101 }, (_, i) => ({ id: `cv-${i}`, full_name: `Candidate ${i}`,
    file_name: "cv.pdf", status: "ready", skills: [], years_experience: null }));
  const { context, element } = frontend({ "/cvs": rows });
  await vm.runInContext("loadCVs()", context);
  assert.equal((element("#cv-rows").innerHTML.match(/data-view=/g) || []).length, 100);
  assert.equal(element("#cv-next").disabled, false);
  assert.equal(element("#cv-prev").disabled, true);
  assert.equal(element("#cv-page").textContent, "CV 1–100");
});
