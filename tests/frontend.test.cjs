const test = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");
const fs = require("node:fs");
const path = require("node:path");

function frontend(responses = {}) {
  const elements = new Map();
  const requests = [];
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
    fetch: async (url, opts = {}) => {
      requests.push({ url, ...opts });
      return { status: 200, ok: true, json: async () => responses[url] || responses[url.split("?")[0]] || [] };
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../src/frontend/app.js"), "utf8"), context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../src/frontend/corpus.js"), "utf8"), context);
  return { context, element, history, requests };
}

test("CV list opens a detail action and escapes names", async () => {
  const { context, element } = frontend({ "/cvs": [{ id: "cv-a", full_name: "<Candidate>",
    file_name: "cv.pdf", status: "ready", skills: ["Python"], years_experience: null }] });
  await vm.runInContext("loadCVs()", context);
  assert.match(element("#cv-rows").innerHTML, /data-view="cv-a"/);
  assert.match(element("#cv-rows").innerHTML, /&lt;Candidate&gt;/);
});

test("corpus console restores saved jobs, escapes preview and does not start work on read", async () => {
  const { context, element, requests } = frontend({
    "/corpus": { corpora: [{id:"sample",records:20,next_record:5}], application_counts:{ready:5},source_available:true,jobs:[] },
    "/corpus/sample/preview": {rows:[{record:1,name:"<Candidate>",status:"not_processed",excerpt:"<script>"}],offset:0,total:20},
  });
  await vm.runInContext("loadCorpus()",context);
  assert.match(element("#corpus-summary").innerHTML,/Hồ sơ trong tập đang chọn/);
  assert.match(element("#corpus-preview").innerHTML,/&lt;Candidate&gt;/);
  assert.doesNotMatch(element("#corpus-preview").innerHTML,/<script>/);
  assert.equal(element("#corpus-next").disabled,false);
  assert.equal(requests.filter(r => r.method === "POST").length,0);
});

test("corpus ingest submits numeric batch selection through API", async () => {
  const {context,element,requests} = frontend({
    "/corpus/jobs": {id:"job1"},
    "/corpus": {corpora:[{id:"sample",records:20,next_record:0}],source_available:true,application_counts:{},jobs:[]},
  });
  element("#corpus-select").value="sample";
  element("#corpus-limit").value="10";
  element("#corpus-batch-size").value="5";
  element("#corpus-mode").value="range";
  element("#corpus-offset").value="21";
  await vm.runInContext("submitCorpus('ingest')",context);
  const post=requests.find(r=>r.url==="/corpus/jobs");
  assert.deepEqual(JSON.parse(post.body),{action:"ingest",corpus_id:"sample",limit:10,batch_size:5,mode:"range",offset:20});
});

test("corpus benchmark shows measured latency and keeps its scope visible", () => {
  const {context,element} = frontend();
  vm.runInContext(`renderCorpusJob({id:'abc',action:'benchmark',status:'completed',phase:'Đã hoàn tất',
    processed:0,total:20,progress:20,result:{records:20,query_repeats:2,build_seconds:1,
    queries:[{query:'Python AND FastAPI',matches:3,p50_ms:0.4,p95_ms:0.7}]}})`,context);
  assert.match(element("#corpus-job").innerHTML,/0.7/);
  assert.match(element("#corpus-job").innerHTML,/không gồm độ trễ LLM/);
  assert.match(element("#corpus-job").innerHTML,/corpus\/jobs\/abc\/report/);
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


test("soft ranking and completed-with-errors are visible even with partial questions", async () => {
  const { context, element } = frontend({ "/searches/s": {
    id:"s", query:"dev 3 years",top_k:1,status:"completed",outcome:"completed_with_errors",
    question_count:1,expected_question_count:6,errors:[{node:"explain",code:"TimeoutError"}],criteria:{},
    questions:[{cv_id:"a",category:"technical",question:"Why?"}],
    shortlist:[{cv_id:"a",full_name:"A",score:.9,rank:1,approved:true,matched_skills:[],missing_skills:[]}]
  }});
  await vm.runInContext("openSearch('s')",context);
  const html=element("#search-view").innerHTML;
  assert.match(html,/điểm cao chưa bảo đảm đạt mọi yêu cầu/);
  assert.match(html,/Hoàn tất có lỗi/);
  assert.match(html,/1\/6/);
  assert.match(html,/TimeoutError/);
});
