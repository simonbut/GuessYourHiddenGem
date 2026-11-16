/* Dream Game Finder - the recommender, running in the visitor's browser.

   The static site (build_site.py) has no server. This module worker loads
   Pyodide (Python compiled to WebAssembly) from the jsDelivr CDN, the same
   recommend.py / quiz.py / server.py the local server uses, and a slimmed
   model, then answers the page's API calls with webapi.handle().

   Round 1 of the quiz and the site status are precomputed, so the page is
   usable while Python and the data (~35 MB on the first visit, cached after
   that) are still downloading. */
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
const PY_FILES = ["recommend.py", "quiz.py", "server.py", "webapi.py"];
const V = new URL(self.location.href).searchParams.get("v") || "";
const at = (p) => new URL(`${p}?v=${encodeURIComponent(V)}`, self.location.href).href;
const post = (m) => self.postMessage(m);

async function getJSON(path) {
  const r = await fetch(at(path));
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

// download with byte progress
async function getBytes(path, onBytes) {
  const r = await fetch(at(path));
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  if (!r.body || !r.body.getReader) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    onBytes(got);
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

// --- quick things first: status + precomputed first rounds (optional) ---
const statusP = getJSON("data/status.json");
const rounds1P = getJSON("data/rounds1.json").catch(() => ({}));

// --- then the real engine ---
const progress = { data: 0, dataTotal: 0, python: false };
function report(stage) {
  const pct = progress.dataTotal ? Math.min(1, progress.data / progress.dataTotal) : 0;
  post({ type: "progress", stage, data: pct, python: progress.python });
}

let handle = null;
const ready = (async () => {
  const status = await statusP;
  const sizes = status.data_bytes || {};
  progress.dataTotal = (sizes["model.npz"] || 0) + (sizes["meta.json.gz"] || 0);
  const got = { model: 0, meta: 0 };
  const onBytes = (k) => (n) => { got[k] = n; progress.data = got.model + got.meta; report("data"); };
  // imported here, not at the top, so status and round 1 never wait for the CDN
  const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
  const [pyodide, model, meta, ...sources] = await Promise.all([
    loadPyodide({ indexURL: PYODIDE_URL, packages: ["numpy", "scipy"] })
      .then((py) => { progress.python = true; report("python"); return py; }),
    getBytes("data/model.npz", onBytes("model")),
    getBytes("data/meta.json.gz", onBytes("meta")),
    ...PY_FILES.map((f) => fetch(at(`py/${f}`)).then((r) => {
      if (!r.ok) throw new Error(`py/${f}: HTTP ${r.status}`);
      return r.text();
    })),
  ]);
  report("starting");
  pyodide.FS.mkdirTree("/app/data");
  PY_FILES.forEach((f, i) => pyodide.FS.writeFile(`/app/${f}`, sources[i]));
  pyodide.FS.writeFile("/app/data/model.npz", model);
  // a host that adds "Content-Encoding: gzip" makes the browser unzip it already
  const gz = meta[0] === 0x1f && meta[1] === 0x8b;
  const metaPath = gz ? "/app/data/meta.json.gz" : "/app/data/meta.json";
  pyodide.FS.writeFile(metaPath, meta);
  pyodide.runPython(`
import sys
sys.path.insert(0, "/app")
import webapi
webapi.load("/app/data/model.npz", "${metaPath}")
`);
  handle = pyodide.globals.get("webapi").handle;
  post({ type: "ready" });
})();
ready.catch((e) => post({ type: "failed", error: String(e && e.message || e) }));

self.onmessage = async (e) => {
  const { id, method, path, body } = e.data || {};
  const reply = (status, data) => post({ id, status, data });
  try {
    if (path === "/api/status") return reply(200, await statusP);
    // round 1 is ready-made - unless they've added favourites, which change it
    const extra = (body && body.extra) || {};
    const plain = !["favorites", "more", "less"].some((k) => Array.isArray(extra[k]) && extra[k].length);
    if (path === "/api/quiz/next" && !handle && plain && body && Array.isArray(body.history) && !body.history.length) {
      const first = (await rounds1P)[String(body.seed)];
      if (first) return reply(200, first);       // same answer Python gives for this seed
    }
    await ready;
    const out = JSON.parse(handle(method, path, body === undefined ? "" : JSON.stringify(body)));
    reply(out.status, out.data);
  } catch (err) {
    reply(503, { error: `The recommender couldn't start in this browser (${String(err && err.message || err)}). ` +
      "Try reloading the page, or another browser." });
  }
};
