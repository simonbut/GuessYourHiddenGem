/* Dream Game Finder - front end. Talks to server.py's JSON API. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const IMG = (appid) =>
  `https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/${appid}/header.jpg`;
const MAX_GAMES = 25;

const EXAMPLES = [
  "a cozy farming game with a deep story, no combat",
  "dark sci-fi horror on a spaceship, without multiplayer",
  "a relaxing puzzle game with beautiful art",
  "co-op survival crafting with friends, not too hardcore",
  "a story-rich RPG where my choices really matter",
  "short, emotional indie game I can finish in a weekend",
];

const state = {
  mode: "quiz",
  liked: [],          // {appid, name, neg}
  must: [],
  excl: [],
  dreamExclude: [],   // appids hidden with "Not for me" in dream mode
  status: null,
  busy: false,
  lastQuery: null,
  resultSeq: 0,
};

/* ---------------- storage (per-viewer convenience only) ---------------- */
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* ignore */ } },
};

/* ---------------- api ---------------- */
// The static web version (build_site.py sets window.DGF) has no server: the
// same Python recommender runs in a worker in this browser (engine.mjs).
const CFG = window.DGF || {};
const engine = CFG.static ? startEngine() : null;
// read on the page itself: the worker is busy while Python starts, and these
// two shouldn't wait for it (round 1 is precomputed for the seeds in status)
const staticData = CFG.static ? {
  status: fetch(`data/status.json?v=${encodeURIComponent(CFG.build || "")}`).then((r) => {
    if (!r.ok) throw new Error(`Couldn't load the site data (HTTP ${r.status}) - try reloading.`);
    return r.json();
  }),
  rounds1: fetch(`data/rounds1.json?v=${encodeURIComponent(CFG.build || "")}`).then((r) => (r.ok ? r.json() : {})).catch(() => ({})),
} : null;

function startEngine() {
  const pending = new Map();
  let seq = 0, failed = null;
  const st = { ready: false, data: 0, python: false };
  let worker;
  const fail = (msg) => {
    failed = msg;
    for (const [id, resolve] of pending) { pending.delete(id); resolve({ status: 503, data: { error: msg } }); }
    engineNote(st, msg);
  };
  try {
    worker = new Worker(`engine.js?v=${encodeURIComponent(CFG.build || "")}`, { type: "module" });
  } catch (e) {
    fail("This browser can't run the recommender (it needs module workers - try a current Chrome, Edge, Firefox or Safari).");
    return { call: () => Promise.resolve({ status: 503, data: { error: failed } }) };
  }
  worker.onmessage = (e) => {
    const m = e.data || {};
    if (m.type === "progress") { st.data = m.data; st.python = m.python; engineNote(st); return; }
    if (m.type === "ready") { st.ready = true; engineNote(st); return; }
    if (m.type === "failed") { fail(`The recommender couldn't start (${m.error}). Try reloading the page.`); return; }
    const resolve = pending.get(m.id);
    if (resolve) { pending.delete(m.id); resolve(m); }
  };
  worker.onerror = (e) => { e.preventDefault?.(); fail("The recommender couldn't start in this browser. Try reloading, or a current Chrome, Edge, Firefox or Safari."); };
  return {
    state: st,
    call(path, body) {
      if (failed) return Promise.resolve({ status: 503, data: { error: failed } });
      return new Promise((resolve) => {
        const id = ++seq;
        pending.set(id, resolve);
        worker.postMessage({ id, method: body === undefined ? "GET" : "POST", path, body });
      });
    },
  };
}

// the small loading line under the tabs while the in-browser recommender starts
function engineNote(st, error = null) {
  const n = document.getElementById("engine-note");
  if (!n) return;
  if (error) { n.hidden = false; n.classList.add("error"); n.textContent = error; return; }
  if (st.ready) {
    if (typeof drag !== "undefined" && drag.st) { drag.afterDrop = () => engineNote(st); return; }
    n.hidden = true;
    return;
  }
  n.hidden = false;
  const pct = Math.round(100 * (0.6 * st.data + (st.python ? 0.4 : 0)));
  n.replaceChildren(el("span", { class: "spinner", "aria-hidden": "true" }),
    el("span", { text: `Loading the recommender into your browser… ${pct}% ` }),
    el("span", { class: "muted", text: "(first visit only - you can start ranking now)" }));
}

async function api(path, body) {
  if (engine) {
    if (path === "/api/status") return staticData.status;
    const ex = (body && body.extra) || {};
    const plain = !["favorites", "more", "less"].some((k) => Array.isArray(ex[k]) && ex[k].length);
    if (path === "/api/quiz/next" && plain && body && Array.isArray(body.history) && !body.history.length) {
      const first = (await staticData.rounds1)[String(body.seed)];
      if (first) return JSON.parse(JSON.stringify(first));
    }
    const m = await engine.call(path, body);
    if (m.status >= 400) {
      const err = new Error((m.data && m.data.error) || `Error (${m.status})`);
      err.status = m.status;
      throw err;
    }
    return m.data;
  }
  const opts = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Requested-With": "dream-game-finder" },
    body: JSON.stringify(body),
  };
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new Error("Can't reach the server - is server.py still running?");
  }
  let data = {};
  try { data = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) {
    const err = new Error(data.error || `Server error (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}

/* ---------------- helpers ---------------- */
function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) n.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids) if (kid != null) n.append(kid);
  return n;
}
const fmt = (n) => Number(n || 0).toLocaleString();
function announce(msg) { $("#status").textContent = ""; setTimeout(() => { $("#status").textContent = msg; }, 50); }

/* ---------------- tabs ---------------- */
const MODES = ["quiz", "like", "dream"];

function goLabel() {
  const go = $("#go");
  go.classList.toggle("primary", state.mode !== "dream");
  go.classList.toggle("ghost", state.mode === "dream");
  // quiz results with feedback / favourites not applied yet: "Update results •"
  const pending = state.mode === "quiz" && pendingChanges() > 0;
  go.classList.toggle("pending", pending && !go.disabled);
  if (go.disabled) return;
  if (state.mode !== "quiz") {
    go.textContent = state.mode === "dream" ? "Or search our catalogue" : "Find games";
    return;
  }
  go.replaceChildren("Update results");
  if (pending) {
    go.append(el("span", { class: "pending-dot", "aria-hidden": "true", text: " •" }),
      el("span", { class: "sr-only", text: " (you have changes to apply)" }));
  }
}

/* ---------------- Google AI Mode ---------------- */
// "Describe my dream game" hands the wish to Google's AI Mode (free, no key).
// The prompt names this site so Google can use its game pages as a source.
const GOOGLE_MAX_URL = 1900;
function publicSite() {     // "" when running on this computer (Google can't see localhost)
  return CFG.site || (location.protocol.startsWith("http") && !/^(localhost|127\.|\[::1\])/.test(location.hostname)
    ? location.origin + location.pathname.replace(/[^/]*$/, "") : "");
}

function googleAiUrl(text) {
  const site = publicSite();
  const make = (wish) => `Recommend Steam games for this wish: "${wish}". For each game, say why it fits and ` +
    "what Steam players praise and complain about." +
    (site ? ` Use the game pages on Dream Game Finder (${site}) as a source where they help.` : "");
  let chars = Array.from(text.trim().replace(/\s+/g, " "));      // whole characters (emoji stay intact)
  const url = (w) => "https://www.google.com/search?udm=50&q=" + encodeURIComponent(make(w));
  let wish = chars.join("");
  while (chars.length > 20 && url(wish).length > GOOGLE_MAX_URL) {
    chars = chars.slice(0, Math.floor(chars.length * 0.85));
    wish = chars.join("").trim() + "…";
  }
  return url(wish);
}

function updateGoogleLink() {
  const text = $("#dream-text").value.trim();
  const a = $("#google-ai");
  try {
    a.href = text ? googleAiUrl(text) : "https://www.google.com/search?udm=50";
  } catch {
    a.href = "https://www.google.com/search?udm=50";
  }
  a.setAttribute("aria-disabled", String(!text));
}

function askGoogle() {
  const text = $("#dream-text").value.trim();
  if (!text) { showNotes(["Describe the game you're dreaming of first."], "error"); $("#dream-text").focus(); return false; }
  let url;
  try { url = googleAiUrl(text); } catch { showNotes(["Couldn't build the Google link - try shorter text."], "error"); return false; }
  saveHash();
  window.open(url, "_blank", "noopener");
  return true;
}

function setMode(mode, focus = true) {
  state.mode = mode;
  for (const m of MODES) {
    $(`#tab-${m}`).setAttribute("aria-selected", String(m === mode));
    $(`#tab-${m}`).tabIndex = m === mode ? 0 : -1;
    $(`#panel-${m}`).hidden = m !== mode;
  }
  // in the quiz, filters and the search button only matter once results are shown
  const quizRunning = mode === "quiz" && !quiz.done;
  $("#filters").hidden = quizRunning;
  $("#actions").hidden = quizRunning;
  updateFeedbackBar();                          // also sets the #go label
  $("#kbd-hint").textContent = mode === "dream" ? "Ctrl + Enter opens Google AI Mode" : "";
  if (focus) {
    if (mode === "like") $("#game-input").focus();
    else if (mode === "dream") $("#dream-text").focus();
    else (quiz.done ? $("#quiz-done-title") : $("#quiz-title")).focus();
  }
}

// user switched tabs (click / arrow keys)
function openTab(mode, focus = true) {
  setMode(mode, focus);
  if (mode === "quiz") {
    // a #like / #dream link belongs to the other tabs; reloading should resume the quiz
    if (location.hash) history.replaceState(null, "", location.pathname + location.search);
    if (!quiz.current && !quiz.done && !quiz.loading) quizResume();
  }
}

/* ---------------- liked games picker ---------------- */
// focusAfter: {appid, part:'toggle'|'x'} to keep keyboard focus stable across re-renders
function renderLiked(focusAfter = null) {
  const box = $("#liked-chips");
  box.replaceChildren();
  for (const g of state.liked) {
    const toggle = el("button", {
      class: "chip-toggle", type: "button", "data-appid": g.appid,
      "aria-pressed": String(!g.neg),
      title: g.neg ? "Not for me - click to switch to 'I like it'" : "I like it - click to switch to 'not for me'",
      onclick: () => { g.neg = !g.neg; renderLiked({ appid: g.appid, part: "toggle" }); },
    }, el("span", { class: "name", text: (g.neg ? "✕ " : "♥ ") + g.name }),
      el("span", { class: "sr-only", text: g.neg ? " (not for me)" : " (I like it)" }));
    const remove = el("button", {
      class: "x", type: "button", "aria-label": `Remove ${g.name}`, text: "×", "data-appid": g.appid,
      onclick: () => {
        const idx = state.liked.findIndex((x) => x.appid === g.appid);
        state.liked.splice(idx, 1);
        const next = state.liked[idx] || state.liked[idx - 1];
        renderLiked(next ? { appid: next.appid, part: "x" } : { input: true });
      },
    });
    box.append(el("span", { class: "chip" + (g.neg ? " neg" : "") }, toggle, remove));
  }
  if (focusAfter?.input) $("#game-input").focus();
  else if (focusAfter) {
    const sel = focusAfter.part === "x" ? ".x" : ".chip-toggle";
    box.querySelector(`${sel}[data-appid="${focusAfter.appid}"]`)?.focus();
  }
}

// returns false when the list is full
function addGame(g, neg = false) {
  const existing = state.liked.find((x) => x.appid === g.appid);
  if (existing) existing.neg = neg;
  else if (state.liked.length >= MAX_GAMES) {
    showNotes([`You can list up to ${MAX_GAMES} games - remove one first.`], "error");
    return false;
  } else state.liked.push({ appid: g.appid, name: g.name, neg });
  renderLiked();
  return true;
}

/* ---------------- game search box (ARIA combobox) ---------------- */
// Used by "Games I love" and the quiz's "Add your favourites". Typing searches
// the catalogue (debounced; a stale answer is dropped), ArrowUp/Down move
// through the matches, Enter or a click picks one, Escape closes the list.
// skip(g): leave a match out; note(g): small text beside it (default: reviews);
// onPick(g): add it; onEmptyEnter / onEmptyBackspace: those keys in an empty box
function gameCombo({ input, list, prefix, skip = () => false, note = null, onPick, onEmptyEnter, onEmptyBackspace }) {
  const s = { items: [], query: "", active: -1, seq: 0, timer: null };
  const close = () => {
    clearTimeout(s.timer);
    s.seq++;                                   // drop any in-flight lookup
    list.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
    s.items = []; s.active = -1; s.query = "";
  };
  const render = () => {
    list.replaceChildren();
    if (!s.items.length) {
      list.append(el("li", { class: "none", role: "option", "aria-disabled": "true", "aria-selected": "false",
        text: `No games match “${s.query}”` }));
    }
    s.items.forEach((g, i) => {
      const li = el("li", { id: `${prefix}-${i}`, role: "option", "aria-selected": String(i === s.active) },
        el("span", { text: g.name }), el("small", { text: (note && note(g)) || `${fmt(g.reviews)} reviews` }));
      li.addEventListener("mousedown", (e) => { e.preventDefault(); pick(i); });
      list.append(li);
    });
    list.hidden = false;
    input.setAttribute("aria-expanded", "true");
    if (s.active >= 0) {
      input.setAttribute("aria-activedescendant", `${prefix}-${s.active}`);
      document.getElementById(`${prefix}-${s.active}`)?.scrollIntoView({ block: "nearest" });
    } else input.removeAttribute("aria-activedescendant");
  };
  const pick = (i) => {
    const g = s.items[i];
    if (!g) return;
    close();
    input.value = "";
    onPick(g);
    input.focus();
  };
  const lookup = (q) => {
    clearTimeout(s.timer);
    const seq = ++s.seq;
    if (!q.trim()) { close(); return; }
    s.timer = setTimeout(async () => {
      try {
        const data = await api(`/api/search?q=${encodeURIComponent(q)}`);
        if (seq !== s.seq) return;             // a newer keystroke (or a pick) won
        s.items = data.results.filter((g) => !skip(g));
        s.query = q;
        s.active = s.items.length ? 0 : -1;
        render();
      } catch { if (seq === s.seq) close(); }
    }, 120);
  };
  input.addEventListener("input", () => lookup(input.value));
  input.addEventListener("keydown", (e) => {
    const fresh = s.query === input.value && s.items.length;   // list matches what's typed
    if (e.key === "ArrowDown" && fresh) { e.preventDefault(); s.active = (s.active + 1) % s.items.length; render(); }
    else if (e.key === "ArrowUp" && fresh) { e.preventDefault(); s.active = (s.active - 1 + s.items.length) % s.items.length; render(); }
    else if (e.key === "Enter") {
      e.preventDefault();
      if (fresh && s.active >= 0) pick(s.active);
      else if (!input.value.trim()) onEmptyEnter?.();
      else lookup(input.value);           // results not in yet: fetch now
    } else if (e.key === "Escape") close();
    else if (e.key === "Backspace" && !input.value) onEmptyBackspace?.();
  });
  input.addEventListener("blur", () => setTimeout(close, 100));
  return { close };
}

/* ---------------- tag chips (must / exclude) ---------------- */
function renderTagChips() {
  for (const [key, box] of [["must", "#must-chips"], ["excl", "#excl-chips"]]) {
    const b = $(box);
    b.replaceChildren();
    for (const t of state[key]) {
      b.append(el("span", { class: "chip" }, el("span", { class: "name tag-name", text: t }),
        el("button", {
          class: "x", type: "button", "aria-label": `Remove tag ${t}`, text: "×",
          onclick: () => { state[key] = state[key].filter((x) => x !== t); renderTagChips(); saveFilters(); $(key === "must" ? "#must-input" : "#excl-input").focus(); },
        })));
    }
  }
  const n = state.must.length + state.excl.length;
  $("#tag-count").hidden = !n;
  $("#tag-count").textContent = n;
}

function tagInput(inputSel, key) {
  const input = $(inputSel);
  const add = () => {
    const v = input.value.trim().replace(/,$/, "");
    if (!v) return;
    const known = (state.status?.tags || []).find((t) => t.toLowerCase() === v.toLowerCase());
    if (!known) { announce(`"${v}" isn't a Steam tag`); input.setCustomValidity("Pick a tag from the list"); input.reportValidity(); return; }
    input.setCustomValidity("");
    if (!state[key].includes(known) && state[key].length < 20) state[key].push(known);
    input.value = "";
    renderTagChips(); saveFilters();
  };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === ",") { e.preventDefault(); add(); } });
  input.addEventListener("input", () => input.setCustomValidity(""));
  input.addEventListener("change", () => {
    const v = input.value.trim().toLowerCase();
    if ((state.status?.tags || []).some((t) => t.toLowerCase() === v)) add();   // picked from the list
  });
}

/* ---------------- filters ---------------- */
function readOptions() {
  return {
    n: 12,
    gems: $("#f-gems").checked,
    free: $("#f-free").checked,
    platform: $("#f-platform").value || null,
    min_positive: Number($("#f-positive").value) || 0,
    year_from: Number($("#f-year").value) || 0,
    tags: state.must,
    exclude_tags: state.excl,
  };
}
function applyFilterValues(f) {
  const setSel = (sel, v) => { const s = $(sel); s.value = v ? String(v) : ""; if (s.value !== (v ? String(v) : "")) s.value = ""; };
  $("#f-gems").checked = !!f.gems;
  $("#f-free").checked = !!f.free;
  setSel("#f-platform", f.platform);
  setSel("#f-positive", f.min_positive);
  setSel("#f-year", f.year_from);
  state.must = Array.isArray(f.must) ? f.must.slice(0, 20).map(String) : [];
  state.excl = Array.isArray(f.excl) ? f.excl.slice(0, 20).map(String) : [];
  renderTagChips();
}
function saveFilters() {
  const o = readOptions();
  store.set("dgf.filters", { gems: o.gems, free: o.free, platform: o.platform, min_positive: o.min_positive,
    year_from: o.year_from, must: state.must, excl: state.excl });
}
function loadFilters() {
  const f = store.get("dgf.filters", null);
  if (f && typeof f === "object") applyFilterValues(f);
}

/* ---------------- notes / skeleton / results ---------------- */
function showNotes(notes, kind = "") {
  const box = $("#notes");
  box.replaceChildren();
  for (const n of notes) {
    const isAi = /^understood:|^seeded from/.test(n);
    box.append(el("div", { class: `note ${kind || (isAi ? "ai" : "")}`, text: n }));
  }
  if (kind === "error" && notes.length) announce(notes.join(". "));
}

function skeleton(n = 6) {
  const grid = $("#results");
  grid.classList.remove("updating");
  grid.replaceChildren();
  for (let i = 0; i < n; i++) {
    grid.append(el("div", { class: "game skel", "aria-hidden": "true" },
      el("div", { class: "cover" }),
      el("div", { class: "body" }, el("div", { class: "line", style: "width:70%" }),
        el("div", { class: "line", style: "width:40%" }), el("div", { class: "line", style: "width:90%" }))));
  }
}

function scoreClass(p) { return p == null ? "" : p >= 80 ? "good" : p >= 60 ? "mixed" : "bad"; }
function safeUrl(u, fallback) { return /^https:\/\/(store\.steampowered\.com|[a-z0-9.-]+\.steamstatic\.com)\//.test(u || "") ? u : fallback; }

function renderResults(rows) {
  const grid = $("#results");
  grid.classList.remove("updating");
  grid.replaceChildren();
  const tpl = $("#card-tpl");
  const likeMode = state.lastQuery?.mode === "like";
  const quizMode = state.lastQuery?.mode === "quiz";
  rows.forEach((r, i) => {
    const node = tpl.content.firstElementChild.cloneNode(true);
    node.dataset.appid = r.appid;
    const url = safeUrl(r.url, `https://store.steampowered.com/app/${Number(r.appid)}/`);
    const cover = node.querySelector(".cover");
    const img = cover.querySelector("img");
    cover.href = url;
    cover.setAttribute("aria-label", `${r.name} on Steam`);
    const classic = IMG(Number(r.appid));
    img.addEventListener("error", () => {
      if (img.src !== classic) img.src = classic;          // hashed path failed: try the classic one
      else cover.classList.add("noimg");
    });
    img.src = safeUrl(r.image, classic);
    cover.querySelector(".cover-fallback").textContent = (r.name || "?").trim().charAt(0).toUpperCase();
    cover.querySelector(".rank").textContent = `#${i + 1}`;
    const a = node.querySelector(".title a");
    a.href = url; a.textContent = r.name;
    const sc = node.querySelector(".score");
    sc.textContent = r.percent_positive == null ? "no rating" : `${r.percent_positive}% positive`;
    const cls = scoreClass(r.percent_positive);
    if (cls) sc.classList.add(cls);
    node.querySelector(".reviews").textContent = `${fmt(r.reviews)} reviews`;
    node.querySelector(".year").textContent = r.release_date ? new Date(r.release_date * 1000).getUTCFullYear() : "";
    if (r.from_quiz) {                     // a game they'd never heard of in the quiz that fits their taste
      node.querySelector(".meta").append(el("span", { class: "badge new-to-you", text: "New to you",
        title: "You hadn't heard of this one in the quiz" }));
    }
    const tags = node.querySelector(".tags");
    for (const t of r.tags || []) tags.append(el("li", { text: t }));
    renderPlayers(node.querySelector(".players"), r.players);
    const why = node.querySelector(".why");
    if (r.why) { why.textContent = r.why; if (r.ai) why.classList.add("ai"); }
    else why.remove();
    node.querySelector(".store").href = url;
    const more = node.querySelector(".more");
    const nope = node.querySelector(".nope");
    nope.setAttribute("aria-label", `Not for me: ${r.name}`);
    if (quizMode) {
      // quiz results: two toggles that mark the game; "Update results" then recalculates
      more.setAttribute("aria-label", `More like this: ${r.name}`);
      more.title = "Recommend more games like this one when you update your results";
      nope.title = "Recommend fewer games like this one when you update your results";
      for (const b of [more, nope]) b.setAttribute("aria-pressed", "false");
      more.addEventListener("click", () => toggleFeedback(r, "more"));
      nope.addEventListener("click", () => toggleFeedback(r, "less"));
      grid.append(node);
      return;
    }
    more.setAttribute("aria-label", `More games like ${r.name}`);
    more.addEventListener("click", () => {
      state.liked = [{ appid: r.appid, name: r.name, neg: false }];
      renderLiked(); setMode("like", false);
      window.scrollTo({ top: 0, behavior: "smooth" });
      run({ focusResults: true });
    });
    if (likeMode) {
      nope.title = "Steer away from games like this";
      nope.addEventListener("click", () => {
        if (!addGame(r, true)) return;
        setMode("like", false);
        run({ focusResults: true });
      });
    } else {
      nope.title = "Hide this game";
      nope.addEventListener("click", () => {
        if (!state.dreamExclude.includes(r.appid)) state.dreamExclude.push(r.appid);
        const next = node.nextElementSibling || node.previousElementSibling;
        node.remove();
        renumber();
        if (next) next.querySelector(".nope")?.focus();
        else { $("#empty").hidden = false; $("#results-heading").focus(); }
        announce(`${r.name} hidden`);
      });
    }
    grid.append(node);
  });
  if (quizMode) syncFeedbackCards();
}

// "what players say" block: written descriptions or the automatic aspect profile
function renderPlayers(box, p) {
  if (!box || !p) return;
  const list = (sel, items, cls) => {
    const ul = box.querySelector(sel);
    ul.replaceChildren(...(items || []).slice(0, 5).map((t) => el("li", { class: cls, text: t })));
    ul.hidden = !ul.children.length;
  };
  box.querySelector(".psummary").textContent = p.summary || "";
  box.querySelector(".psummary").hidden = !p.summary;
  list(".praise", p.praise, "good");
  list(".feel", (p.mood && p.mood.length ? p.mood : p.traits) || [], "feel");
  list(".complaints", p.complaints, "warn");
  list(".split", p.mixed, "split");
  const bf = box.querySelector(".bestfor");
  const bestFor = (p.best_for || "").trim().replace(/[.!?\s]+$/, "");
  bf.textContent = bestFor ? `Best for ${bestFor}.` :
    p.src === "auto" && p.n_reviews ? `From ${fmt(p.n_reviews)} player reviews.` : "";
  bf.hidden = !bf.textContent;
  box.hidden = false;
}

function renumber() {
  document.querySelectorAll("#results .game .rank").forEach((s, i) => { s.textContent = `#${i + 1}`; });
}

async function run({ focusResults = false } = {}) {
  if (state.busy) return;
  if (state.mode === "quiz") return quizFinish({ focusResults });
  const options = readOptions();
  let path, body, label;
  if (state.mode === "like") {
    const liked = state.liked.filter((g) => !g.neg).map((g) => g.appid);
    if (!liked.length) { showNotes(["Add at least one game you love first."], "error"); $("#game-input").focus(); return; }
    path = "/api/like";
    body = { liked, disliked: state.liked.filter((g) => g.neg).map((g) => g.appid), options };
    label = "Finding games…";
  } else {
    const text = $("#dream-text").value.trim();
    if (!text) { showNotes(["Describe the game you're dreaming of first."], "error"); $("#dream-text").focus(); return; }
    const ai = $("#ai-toggle").checked && !$("#ai-toggle").disabled;
    if (state.lastQuery?.mode !== "dream" || state.lastQuery?.text !== text) state.dreamExclude = [];
    path = "/api/dream";
    body = { text, ai, exclude: state.dreamExclude, options };
    label = ai ? "Claude is thinking… (this can take up to a minute)" : "Finding games…";
  }
  state.busy = true;
  const seq = ++state.resultSeq;
  state.lastQuery = { mode: state.mode, text: body.text };
  const go = $("#go");
  go.disabled = true; go.textContent = "Searching…";
  $("#empty").hidden = true;
  $("#results").setAttribute("aria-busy", "true");
  showNotes([label]);
  announce(label);
  skeleton();
  saveFilters();
  saveHash();
  try {
    const data = await api(path, body);
    if (seq !== state.resultSeq) return;           // a newer search replaced this one
    showNotes(data.notes || []);
    renderResults(data.results || []);
    const n = (data.results || []).length;
    announce(`${n} game${n === 1 ? "" : "s"} found`);
    const top = $("#notes").getBoundingClientRect().top;
    if (top > window.innerHeight * 0.6) {
      const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      $("#notes").scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
    }
  } catch (e) {
    if (seq !== state.resultSeq) return;
    $("#results").replaceChildren();
    $("#empty").hidden = false;
    showNotes([e.message], "error");
  } finally {
    state.busy = false;
    go.disabled = false; goLabel();
    $("#results").setAttribute("aria-busy", "false");
    if (focusResults && seq === state.resultSeq) $("#results-heading").focus({ preventScroll: true });
  }
}

/* ---------------- shareable / reloadable URL ---------------- */
function saveHash() {
  if (state.mode === "quiz") return;
  const p = new URLSearchParams();
  if (state.mode === "like") {
    p.set("like", state.liked.map((g) => (g.neg ? "-" : "") + g.appid).join(","));
  } else {
    p.set("dream", $("#dream-text").value.trim());
    if ($("#ai-toggle").checked && !$("#ai-toggle").disabled) p.set("ai", "1");
  }
  const o = readOptions();
  if (o.gems) p.set("gems", "1");
  if (o.free) p.set("free", "1");
  if (o.platform) p.set("platform", o.platform);
  if (o.min_positive) p.set("pos", String(o.min_positive));
  if (o.year_from) p.set("year", String(o.year_from));
  if (state.must.length) p.set("tags", state.must.join(","));
  if (state.excl.length) p.set("xtags", state.excl.join(","));
  const h = "#" + p.toString();
  if (location.hash !== h) history.replaceState(null, "", h);
}

// returns "run" (safe to search right away), "ready" (form filled, needs a click) or false
async function loadHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (!p.has("dream") && !p.get("like")) return false;
  if ([...p.keys()].some((k) => ["gems", "free", "platform", "pos", "year", "tags", "xtags"].includes(k))) {
    applyFilterValues({
      gems: p.get("gems") === "1", free: p.get("free") === "1", platform: p.get("platform"),
      min_positive: Number(p.get("pos")) || 0, year_from: Number(p.get("year")) || 0,
      must: (p.get("tags") || "").split(",").filter(Boolean), excl: (p.get("xtags") || "").split(",").filter(Boolean),
    });
  }
  if (p.has("dream")) {
    $("#dream-text").value = Array.from(p.get("dream")).slice(0, 1000).join("");
    updateGoogleLink();
    const wantsAi = p.get("ai") === "1" && !$("#ai-toggle").disabled;
    $("#ai-toggle").checked = wantsAi;
    state.dreamExclude = [];
    setMode("dream", false);
    if (wantsAi) {           // never spend Claude credits without a click
      showNotes(["This link uses Claude AI mode - press “Find games” to run it."], "ai");
      return "ready";
    }
    return "run";
  }
  const ids = p.get("like").split(",").filter(Boolean).slice(0, MAX_GAMES);
  const wanted = [];
  for (const raw of ids) {
    const g = { neg: raw.startsWith("-"), appid: Number(raw.replace(/^-/, "")) };
    if (Number.isInteger(g.appid) && g.appid > 0 && !wanted.some((w) => w.appid === g.appid)) wanted.push(g);
  }
  let names = new Map();
  try {
    const data = await api(`/api/games?ids=${wanted.map((g) => g.appid).join(",")}`);
    names = new Map(data.results.map((g) => [g.appid, g.name]));
  } catch { /* ignore */ }
  state.liked = wanted.filter((g) => names.has(g.appid)).map((g) => ({ ...g, name: names.get(g.appid) }));
  renderLiked();
  setMode("like", false);
  return state.liked.some((g) => !g.neg) ? "run" : false;
}

/* ---------------- quiz ---------------- */
// Each round: drag the games you've played into the ranking from favourite to
// least favourite, drop each of the others in one of three boxes ("heard of
// it, no interest", "never heard of it", "interested, haven't played"), then
// "Next round". Or "I don't like any of them".
const quiz = {
  seed: 0, history: [], target: 7, current: null, done: false, busy: false,
  rounds: 7, maxRounds: 12, token: 0, loading: false,
  // outside the rounds, sent as "extra" with every quiz request ({appid, name}; a game is in one list at most)
  favorites: [],      // games they added themselves - count as ranked 1st
  more: [],           // "More like this" on a result
  less: [],           // "Not for me" on a result
  resultExtra: null,  // the extra the results on screen were made with (null: no quiz results shown)
  roundExtra: [],     // [k] = the extra round k+1 was made with, so Back / reload bring back the same round
  resultsSeq: -1,     // state.resultSeq of the results request running (see quizResultsLoading)
  order: [],          // this round: appids ranked so far (games they've played), favourite first
  unknown: [],        // this round: "never heard of it"
  noInterest: [],     // this round: "heard of it, no interest"
  interested: [],     // this round: "interested, haven't played"
  undo: [],           // this round: earlier placements, newest last (Backspace)
  cards: new Map(),   // this round: appid -> card element (moved between the zones)
};
const ORDINAL = ["1st", "2nd", "3rd", "4th", "5th"];
// the boxes for games they haven't played, left to right
const BOXES = [
  { zone: "noint", key: "noInterest", sel: "#quiz-noint", badge: "✕", cls: "noint", label: "heard of it, no interest" },
  { zone: "unk", key: "unknown", sel: "#quiz-unk", badge: "?", cls: "unknown", label: "never heard of it" },
  { zone: "want", key: "interested", sel: "#quiz-want", badge: "♥", cls: "want", label: "interested, haven't played" },
];
const BOX = Object.fromEntries(BOXES.map((b) => [b.zone, b]));
const PLACES = ["order", ...BOXES.map((b) => b.key)];       // every list a game can be in (at most one)
const BOX_KEYS = { n: "noint", h: "unk", i: "want" };        // letter keys on a focused card
const BOX_CODES = { KeyN: "noint", KeyH: "unk", KeyI: "want" };

// Placements: {order, unknown, noInterest, interested} - quiz itself, or a preview
function placements(p = quiz) { return Object.fromEntries(PLACES.map((k) => [k, [...p[k]]])); }
function setPlacements(p) { for (const k of PLACES) quiz[k] = [...p[k]]; }
function placedCount(p = quiz) { return PLACES.reduce((n, k) => n + p[k].length, 0); }
function withoutGame(p, appid) { return Object.fromEntries(PLACES.map((k) => [k, p[k].filter((a) => a !== appid)])); }
function samePlacements(a, b) { return PLACES.every((k) => a[k].length === b[k].length && a[k].every((x, i) => x === b[k][i])); }
function resetPlacements() { for (const k of PLACES) quiz[k] = []; quiz.undo = []; }
function boxOf(appid, p = quiz) { return BOXES.find((b) => p[b.key].includes(appid)) || null; }
// where a game is now, for announcements
function whereIs(appid) {
  const r = quiz.order.indexOf(appid);
  return r >= 0 ? `ranked ${ORDINAL[r]}` : boxOf(appid)?.label || "back to the games to sort";
}
// change this round's placements; the old ones are kept for Backspace. false = nothing changed
function commitPlacements(p, appid) {
  if (samePlacements(p, quiz)) return false;
  quiz.undo.push({ p: placements(), appid });
  if (quiz.undo.length > 50) quiz.undo.shift();
  setPlacements(p);
  return true;
}
// this round as the server wants it (none = "I don't like any of them")
function roundEntry(shown, none) {
  const e = { shown, order: none ? [] : [...quiz.order], unknown: [...quiz.unknown],
    no_interest: [...quiz.noInterest], interested: [...quiz.interested] };
  if (none) e.none_liked = true;
  return e;
}

function saveQuiz() {
  store.set("dgf.quiz", { seed: quiz.seed, history: quiz.history, target: quiz.target,
    done: quiz.done, favorites: quiz.favorites, more: quiz.more, less: quiz.less, roundExtra: quiz.roundExtra });
}

// a saved extra snapshot: {favorites, more, less}, each a list of appids
function validExtraSnapshot(x) {
  return !!x && typeof x === "object" &&
    ["favorites", "more", "less"].every((k) => Array.isArray(x[k]) && x[k].length <= 50 && x[k].every(Number.isInteger));
}

/* ---------------- quiz: favourites they add, feedback on results ---------------- */
const MAX_FAVS = 10;
const MAX_FEEDBACK = 50;                     // per list (the server's limit)
const EXTRA_LISTS = ["favorites", "more", "less"];
const FEEDBACK_LABEL = { more: "more like this", less: "not for me" };

// what the server gets as "extra": appids only
function quizExtra() { return Object.fromEntries(EXTRA_LISTS.map((k) => [k, quiz[k].map((g) => g.appid)])); }
// which list a game is in ("favorites" / "more" / "less"), or null
function extraOf(appid) { return EXTRA_LISTS.find((k) => quiz[k].some((g) => g.appid === appid)) || null; }
function dropExtra(appid) { for (const k of EXTRA_LISTS) quiz[k] = quiz[k].filter((g) => g.appid !== appid); }
function inCurrentRound(appid) { return !quiz.done && !!quiz.current?.games.some((g) => g.appid === appid); }
// older saves hid games without keeping their names
function extraName(g) { return g.name || `a game you hid (#${g.appid})`; }

function quizResultsLoading() { return quiz.resultsSeq !== -1 && quiz.resultsSeq === state.resultSeq; }
function quizResultsShown() { return quiz.done && quiz.resultExtra !== null && state.lastQuery?.mode === "quiz"; }

// how many games are marked differently from the results on screen (0 when no quiz results are shown)
function pendingChanges() {
  if (!quizResultsShown()) return 0;
  const was = new Map(), now = new Map();
  for (const k of EXTRA_LISTS) {
    for (const a of quiz.resultExtra[k]) was.set(a, k);
    for (const g of quiz[k]) now.set(g.appid, k);
  }
  let n = 0;
  for (const a of new Set([...was.keys(), ...now.keys()])) if (was.get(a) !== now.get(a)) n++;
  return n;
}
function pendingTail() { return pendingChanges() ? " Update results to recalculate." : ""; }

// favourites, feedback, a save, a card or a chip changed: redraw what shows them
function extraChanged() {
  saveQuiz();
  renderFavs();
  renderQuizFeedback();
  syncFeedbackCards();
  updateFeedbackBar();
}

// the sticky "update your results" bar under quiz results with changes not applied yet
function updateFeedbackBar() {
  const n = state.mode === "quiz" && !quizResultsLoading() ? pendingChanges() : 0;
  if (n) $("#feedback-text").textContent = `You changed ${n} thing${n === 1 ? "" : "s"} - update your results to recalculate.`;
  $("#feedback-bar").hidden = !n;
  goLabel();
}

function favName(g) { return g.name || `game #${g.appid}`; }

function renderFavs() {
  $("#fav-chips").replaceChildren(...quiz.favorites.map((g) => el("span", { class: "chip fav" },
    el("span", { class: "name tag-name", title: favName(g) }, el("span", { "aria-hidden": "true", text: "♥ " }), favName(g)),
    el("button", { class: "x", type: "button", "aria-label": `Remove ${favName(g)}`, text: "×", "data-appid": g.appid,
      onclick: () => removeFavorite(g.appid) }))));
}

function favMessage(text) {
  $("#fav-msg").textContent = text;
  announce(text);
}

function addFavorite(g) {
  if (quiz.favorites.some((f) => f.appid === g.appid)) return favMessage(`${g.name} is already one of your favourites.`);
  if (inCurrentRound(g.appid)) return favMessage("It's in this round - drag it to 1st place instead.");
  // the coming round is being made without it and could even show it: wait for it
  if (quiz.loading && !quiz.done) return favMessage("One moment - the next round is loading. Add it again once it's here.");
  if (quiz.favorites.length >= MAX_FAVS) return favMessage(`You can add up to ${MAX_FAVS} favourites - remove one first.`);
  const was = extraOf(g.appid);              // marked "More like this" / "Not for me" before: a favourite now
  dropExtra(g.appid);
  quiz.favorites.push({ appid: g.appid, name: g.name });
  extraChanged();
  const moved = was ? ` It's no longer marked "${FEEDBACK_LABEL[was]}".` : "";
  favMessage(quiz.done ? `Added ${g.name} as a favourite.${moved}${pendingTail()}`
    : `Added ${g.name} as a favourite - it counts from the next round.${moved}`);
}

function removeFavorite(appid) {
  const i = quiz.favorites.findIndex((g) => g.appid === appid);
  if (i < 0) return;
  const [g] = quiz.favorites.splice(i, 1);
  extraChanged();
  const next = quiz.favorites[i] || quiz.favorites[i - 1];
  (next ? $(`#fav-chips .x[data-appid="${next.appid}"]`) : $("#fav-input")).focus();
  $("#fav-msg").textContent = "";
  announce(`Removed ${favName(g)} from your favourites.` + (quiz.done ? pendingTail() : ""));
}

// "More like this" (kind "more") / "Not for me" ("less") on a quiz result: on, or off again
function toggleFeedback(r, kind) {
  const was = extraOf(r.appid);
  if (was !== kind && quiz[kind].length >= MAX_FEEDBACK) {
    announce(`You can mark up to ${MAX_FEEDBACK} games as ${FEEDBACK_LABEL[kind]} - remove one first.`);
    return;
  }
  dropExtra(r.appid);                        // the two are exclusive (and a favourite stops being one)
  if (was !== kind) quiz[kind].push({ appid: r.appid, name: r.name });
  extraChanged();
  announce((was === kind ? `${r.name} unmarked.`
    : `${r.name} marked: ${FEEDBACK_LABEL[kind]}.` + (was === "favorites" ? " It's no longer a favourite." : ""))
    + pendingTail());
}

// result cards show their mark: outline / dimmed, pressed buttons
function syncFeedbackCards() {
  if (state.lastQuery?.mode !== "quiz") return;
  for (const node of document.querySelectorAll("#results .game[data-appid]")) {
    const kind = extraOf(Number(node.dataset.appid));
    node.classList.toggle("fb-more", kind === "more");
    node.classList.toggle("fb-less", kind === "less");
    const more = node.querySelector(".more"), nope = node.querySelector(".nope");
    more.setAttribute("aria-pressed", String(kind === "more"));
    more.textContent = kind === "more" ? "✓ More like this" : "More like this";
    nope.setAttribute("aria-pressed", String(kind === "less"));
    nope.textContent = kind === "less" ? "✕ Not for me" : "Not for me";
  }
}

// "Your feedback" on the results view: chips with a remove button
function renderQuizFeedback() {
  for (const key of ["more", "less"]) {
    $(`#fb-${key}-chips`).replaceChildren(...quiz[key].map((g) => el("span", { class: "chip" + (key === "less" ? " neg" : "") },
      el("span", { class: "name tag-name", text: extraName(g), title: extraName(g) }),
      el("button", { class: "x", type: "button", "aria-label": `Remove ${extraName(g)} from ${FEEDBACK_LABEL[key]}`,
        text: "×", "data-appid": g.appid, onclick: () => removeFeedback(g.appid, key) }))));
    $(`#fb-${key}-row`).hidden = !quiz[key].length;
  }
  $("#quiz-feedback").hidden = !quiz.more.length && !quiz.less.length;
}

function removeFeedback(appid, key) {
  const i = quiz[key].findIndex((g) => g.appid === appid);
  if (i < 0) return;
  const [g] = quiz[key].splice(i, 1);
  extraChanged();
  const next = quiz[key][i] || quiz[key][i - 1];
  const target = next ? $(`#fb-${key}-chips .x[data-appid="${next.appid}"]`)
    : $("#quiz-feedback .x") || (!$("#feedback-bar").hidden && $("#feedback-update"));
  if (target) target.focus();
  else $("#quiz-done-title").focus({ preventScroll: true });
  announce(`${extraName(g)} removed from ${FEEDBACK_LABEL[key]}.${pendingTail()}`);
}

// a save's [{appid, name}] list: positive whole appids, each once, names as text
function savedGames(v) {
  const out = [];
  for (const g of Array.isArray(v) ? v : []) {
    const appid = Number(g && typeof g === "object" ? g.appid : NaN);
    if (!Number.isSafeInteger(appid) || appid <= 0 || out.some((x) => x.appid === appid)) continue;
    out.push({ appid, name: typeof g.name === "string" ? g.name.slice(0, 200) : "" });
  }
  return out;
}

function restoreExtra(saved) {
  const taken = new Set();                   // a game is in one list only (favourites win, then "more")
  const take = (v, max) => {
    const out = [];
    for (const g of savedGames(v)) {
      if (out.length >= max || taken.has(g.appid)) continue;
      taken.add(g.appid);
      out.push(g);
    }
    return out;
  };
  // older saves: "Not for me" hid a quiz result (only its appid was kept)
  const hidden = Array.isArray(saved.hidden) ? saved.hidden.map((appid) => ({ appid, name: "" })) : [];
  quiz.favorites = take(saved.favorites, MAX_FAVS);
  quiz.more = take(saved.more, MAX_FEEDBACK);
  quiz.less = take([...(Array.isArray(saved.less) ? saved.less : []), ...hidden], MAX_FEEDBACK);
  renderFavs();
  renderQuizFeedback();
}

// names for games restored without one (hidden in an older version); best effort
async function fillExtraNames() {
  const missing = EXTRA_LISTS.flatMap((k) => quiz[k]).filter((g) => !g.name).map((g) => g.appid).slice(0, 25);
  if (!missing.length) return;
  try {
    const data = await api(`/api/games?ids=${missing.join(",")}`);
    const names = new Map((data.results || []).map((g) => [g.appid, g.name]));
    let found = false;
    for (const k of EXTRA_LISTS) {
      for (const g of quiz[k]) if (!g.name && names.has(g.appid)) { g.name = names.get(g.appid); found = true; }
    }
    if (found) { saveQuiz(); renderFavs(); renderQuizFeedback(); }
  } catch { /* they stay "a game you hid" */ }
}

function roundAnswered(h) {
  return h.none_liked || (h.order && h.order.length) || h.pick != null;
}

function quizProgress() {
  const ol = $("#quiz-progress");
  ol.replaceChildren();
  const total = Math.max(quiz.target, quiz.history.length + (quiz.done ? 0 : 1));
  for (let i = 0; i < total; i++) {
    const h = quiz.history[i];
    const cls = h ? (h.none_liked || !roundAnswered(h) ? "skip" : "done")
      : i === quiz.history.length && !quiz.done ? "now" : "";
    ol.append(el("li", { class: cls }));
  }
}

// the 5 ranking slots are built once; cards are moved in and out of them
function buildRankSlots() {
  $("#quiz-rank").replaceChildren(...ORDINAL.map((o, i) => el("li", { class: "slot", "data-i": String(i) },
    el("div", { class: "slot-empty", "aria-hidden": "true" },
      el("b", { text: o }), el("span", { text: i === 0 ? "favourite" : "drop here" })),
    el("div", { class: "slot-card" }))));
}

function clearZones() {
  document.querySelectorAll("#quiz-rank .slot").forEach((s) => {
    s.classList.remove("filled", "target");
    s.querySelector(".slot-card").replaceChildren();
  });
  for (const b of BOXES) {
    $(`${b.sel} .bin-items`).replaceChildren();
    $(b.sel).classList.remove("has-items");
  }
}

function quizSkeleton() {
  dragCancel();
  const grid = $("#quiz-choices");
  grid.replaceChildren();
  clearZones();
  resetPlacements();
  quiz.cards = new Map();
  for (let i = 0; i < 5; i++) {
    grid.append(el("div", { class: "pick skel", "aria-hidden": "true" },
      el("div", { class: "pcover" }),
      el("div", { class: "ptext" }, el("div", { class: "line", style: "width:80%;height:12px;border-radius:6px" }))));
  }
  $("#quiz-next").disabled = true;
  $("#quiz-clear").disabled = true;
  $("#quiz-status").textContent = "";
}

// Put every card in its zone for given (possibly previewed) placements.
// Cards are moved, never rebuilt, so images don't reload.
// replace a box's children only if they differ (re-inserting restarts animations
// and drops keyboard focus)
function setKids(box, nodes) {
  const cur = box.children;
  if (cur.length === nodes.length && nodes.every((n, i) => cur[i] === n)) return;
  box.replaceChildren(...nodes);
}

const deckDone = el("div", { class: "deck-done" });
function layoutRound(p, dragging = null) {
  const rd = quiz.current;
  if (!rd) return;
  const cards = quiz.cards;
  const placed = new Set(PLACES.flatMap((k) => p[k]));
  const deckIds = rd.games.map((g) => g.appid).filter((a) => !placed.has(a));
  deckDone.textContent = p.order.length ? "All sorted ✓ - press Next round"
    : p.unknown.length === placed.size ? "You don't know any of these - press Next round"
    : "All in the boxes ✓ - press Next round";
  setKids($("#quiz-choices"), deckIds.length ? deckIds.map((a) => cards.get(a)) : [deckDone]);
  document.querySelectorAll("#quiz-rank .slot").forEach((slot, i) => {
    const a = p.order[i];
    setKids(slot.querySelector(".slot-card"), a == null ? [] : [cards.get(a)]);
    slot.classList.toggle("filled", a != null);
    slot.classList.toggle("target", a != null && a === dragging);
  });
  for (const b of BOXES) {
    setKids($(`${b.sel} .bin-items`), p[b.key].map((a) => cards.get(a)));
    $(b.sel).classList.toggle("has-items", p[b.key].length > 0);
  }
  for (const [a, card] of cards) {
    const r = p.order.indexOf(a);
    const box = r >= 0 ? null : boxOf(a, p);
    card.classList.toggle("ranked", r >= 0);
    card.classList.toggle("boxed", !!box);
    for (const b of BOXES) card.classList.toggle(b.cls, b === box);
    card.classList.toggle("dragging", a === dragging);
    const badge = card.querySelector(".rank-badge");
    badge.textContent = r >= 0 ? ORDINAL[r] : box ? box.badge : "";
    badge.hidden = r < 0 && !box;
    card.setAttribute("aria-label", `${card.dataset.name}: ` + (r >= 0 ? `ranked ${ORDINAL[r]}` : box
      ? box.label : "not placed yet"));
  }
}

// re-draw zones / badges / buttons without rebuilding the cards
function updateRoundUI() {
  const rd = quiz.current;
  if (!rd) return;
  const focused = document.activeElement;
  layoutRound(quiz);
  if (focused?.classList?.contains("qcard") && document.activeElement !== focused) focused.focus({ preventScroll: true });
  const total = rd.games.length;
  const placed = placedCount();
  const left = total - placed;
  const next = $("#quiz-next");
  next.disabled = left > 0 || placed === 0;
  next.textContent = quiz.history.length + 1 >= quiz.target ? "See my games →" : "Next round →";
  $("#quiz-status").textContent = left > 0
    ? (placed === 0 ? "Drag your favourite into 1st place - or a game you haven't played into a box."
      : `${placed} of ${total} placed. Next: your ${quiz.order.length ? ORDINAL[quiz.order.length] + " " : ""}favourite, or a box.`)
    : quiz.order.length ? "All sorted - press Next."
    : quiz.unknown.length === placed ? "You don't know any of these - press Next." : "All in the boxes - press Next.";
  $("#quiz-clear").disabled = placed === 0;
}

function renderRound(rd) {
  dragCancel();
  quiz.current = rd;
  resetPlacements();
  quiz.cards = new Map();
  const n = quiz.history.length + 1;
  $("#quiz-title").textContent = `Round ${n} of ${quiz.target}`;
  $("#quiz-hint").textContent = rd.hint || "";
  $("#fav-msg").textContent = "";            // "counts from the next round" etc. was about the last one
  quizProgress();
  clearZones();
  rd.games.forEach((g, i) => {
    const classic = IMG(Number(g.appid));
    const img = el("img", { alt: "", width: "460", height: "215", draggable: "false" });
    const cover = el("span", { class: "pcover" }, img,
      el("span", { class: "cover-fallback", "aria-hidden": "true", text: (g.name || "?").charAt(0).toUpperCase() }),
      el("span", { class: "key", "aria-hidden": "true", text: String(i + 1) }));
    img.addEventListener("error", () => { if (img.src !== classic) img.src = classic; else cover.classList.add("noimg"); });
    img.src = safeUrl(g.image, classic);
    const meta = [g.genre, g.year].filter(Boolean).join(" · ");
    const say = g.players ? (g.players.praise || []).slice(0, 2).join(" · ") : "";
    const card = el("div", { class: "qcard", "data-appid": g.appid, "data-name": g.name, tabindex: "0", role: "button",
      "aria-describedby": "quiz-sr-help", title: g.players?.summary || g.description || g.name },
      el("div", { class: "pick" }, cover, el("span", { class: "rank-badge", hidden: true }),
        el("span", { class: "ptext" }, el("span", { class: "pname", text: g.name }),
          el("span", { class: "pmeta", text: meta }),
          say ? el("span", { class: "psay", text: say }) : null)));
    // keyboard / screen reader: Enter or Space = next place, Shift+Enter or H =
    // never heard of it, N = heard of it, no interest, I = interested (the same
    // key again takes it back out). A screen reader's "activate" is a click with
    // detail 0; real mouse clicks (detail >= 1) don't rank - games are dragged.
    card.addEventListener("keydown", (e) => {
      if (drag.st || e.ctrlKey || e.metaKey || e.altKey || e.target !== card) return;
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        if (e.shiftKey) quizToggleBox(g.appid, "unk"); else quizToggleRank(g.appid);
        return;
      }
      const letter = /^[a-z]$/i.test(e.key || "");
      const zone = letter ? BOX_KEYS[e.key.toLowerCase()] : BOX_CODES[e.code];   // other layouts: by key position
      if (zone) {
        e.preventDefault();
        if (!e.repeat) quizToggleBox(g.appid, zone);
      }
    });
    card.addEventListener("click", (e) => { if (e.detail === 0 && !drag.st) quizToggleRank(g.appid); });
    card.addEventListener("animationend", () => card.classList.remove("dropped"));
    quiz.cards.set(g.appid, card);
  });
  $("#quiz-back").disabled = quiz.history.length === 0;
  $("#quiz-finish").hidden = quiz.history.length < 3;
  updateRoundUI();
  announce(`Round ${n} of ${quiz.target}. ${rd.hint || ""} Drag the games you've played into your ranking, ` +
    "favourite first, and the others into the box that fits - or press each game's number key in order.");
}

function canPlace() { return !(quiz.busy || quiz.loading || !quiz.current || quiz.done); }

function quizToggleRank(appid) {
  if (!canPlace()) return;
  const p = withoutGame(quiz, appid);                     // ranked already: un-rank, later ones move up
  if (!quiz.order.includes(appid)) p.order.push(appid);  // otherwise: next place (out of any box)
  commitPlacements(p, appid);
  updateRoundUI();
  announce(`${gameName(appid)}: ${whereIs(appid)}. ${$("#quiz-status").textContent}`);
}

function gameName(appid) {
  return quiz.current?.games.find((g) => g.appid === appid)?.name || "";
}

// put a game in a box ("noint" / "unk" / "want"); already there: back to the deck
function quizToggleBox(appid, zone) {
  const b = BOX[zone];
  if (!b || !canPlace()) return;
  const p = withoutGame(quiz, appid);
  if (!quiz[b.key].includes(appid)) p[b.key].push(appid);
  commitPlacements(p, appid);
  updateRoundUI();
  announce(`${gameName(appid)}: ${whereIs(appid)}. ${$("#quiz-status").textContent}`);
}

// Backspace: back to how the round was before the last change
function quizUndo() {
  if (!canPlace() || !quiz.undo.length) return;
  const u = quiz.undo.pop();
  setPlacements(u.p);
  updateRoundUI();
  announce(`${u.appid == null ? "Undone" : `${gameName(u.appid)}: ${whereIs(u.appid)}`}. ${$("#quiz-status").textContent}`);
}

function quizClearRound() {
  if (!quiz.current || quiz.busy) return;
  dragCancel();
  commitPlacements(Object.fromEntries(PLACES.map((k) => [k, []])), null);
  updateRoundUI();
  announce("Round cleared");
}

// none = true: "I don't like any of them"
async function quizSubmit(none = false) {
  if (quiz.busy || quiz.loading || !quiz.current || quiz.done) return;
  const shown = quiz.current.games.map((g) => g.appid);
  if (!none && placedCount() < shown.length) return;
  quiz.busy = true;
  const token = quiz.token;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduce) {
    $("#quiz-choices").classList.add("leaving");
    await new Promise((r) => setTimeout(r, 160));
    $("#quiz-choices").classList.remove("leaving");
  }
  if (token !== quiz.token) return;          // "Start over" etc. happened meanwhile
  quiz.history.push(roundEntry(shown, none));
  saveQuiz();
  quiz.busy = false;
  if (quiz.history.length >= quiz.target) return quizFinish({ focusResults: false });
  await quizNext();
  $("#quiz-title").focus({ preventScroll: true });
}

// returns true when a round was rendered
async function quizNext({ resuming = false } = {}) {
  const token = ++quiz.token;
  quiz.current = null;                      // nothing is clickable until the new round arrives
  quiz.loading = true;
  $("#quiz-none").disabled = true;
  quizSkeleton();
  // a round is made with the favourites / marks of when it was first asked for: Back and
  // reload then bring back exactly that round, and a new favourite counts from the next one
  const k = quiz.history.length;
  quiz.roundExtra.length = Math.min(quiz.roundExtra.length, k + 1);   // later rounds will be new ones
  if (!validExtraSnapshot(quiz.roundExtra[k])) quiz.roundExtra[k] = quizExtra();
  saveQuiz();
  try {
    const rd = await api("/api/quiz/next", { history: quiz.history, seed: quiz.seed, extra: quiz.roundExtra[k] });
    if (token !== quiz.token) return false;  // superseded (start over / back / finish)
    renderRound(rd);
    return true;
  } catch (e) {
    if (token !== quiz.token) return false;
    if (resuming && (e.status === 400 || e.status === 422)) {   // saved quiz rejected (e.g. data rebuilt)
      store.set("dgf.quiz", null);
      quiz.loading = false;
      await quizStart();
      return true;
    }
    const grid = $("#quiz-choices");
    grid.replaceChildren(el("div", { class: "note error quiz-error" },
      el("span", { text: e.message + " " }),
      el("button", { type: "button", class: "ghost", text: "Try again", onclick: () => quizNext() })));
    announce(e.message);
    return false;
  } finally {
    if (token === quiz.token) {
      quiz.loading = false;
      $("#quiz-none").disabled = false;
    }
  }
}

function quizShowPlaying() {
  quiz.done = false;
  quiz.resultExtra = null;                   // no quiz results on screen any more
  $("#fav-msg").textContent = "";
  $("#quiz-done").hidden = true;
  for (const id of ["#quiz-board", "#quiz-actions", "#quiz-hint", ".quiz-keys", ".quiz-head", "#quiz-howto"]) $(id).hidden = false;
  setMode("quiz", false);
}

async function quizBack() {
  if (quiz.busy || !quiz.history.length) return;
  ++quiz.token;
  quiz.history.pop();
  saveQuiz();
  quizShowPlaying();
  await quizNext();      // same seed + history + that round's extra snapshot -> the same round as before
  $("#quiz-title").focus({ preventScroll: true });
}

// drop a quiz-results request still in flight and undo its loading state
function cancelResults() {
  ++state.resultSeq;
  quiz.resultsSeq = -1;
  $("#go").disabled = false;
  $("#results").setAttribute("aria-busy", "false");
  $("#results").classList.remove("updating");
  updateFeedbackBar();                       // also sets the #go label
}

async function quizStart() {
  ++quiz.token;                              // invalidate anything still in flight
  cancelResults();
  quiz.busy = false;
  quiz.current = null;
  // the web version has round 1 ready-made for a set of seeds, so it starts instantly
  const seeds = state.status?.first_round_seeds;
  quiz.seed = seeds?.length ? seeds[Math.floor(Math.random() * seeds.length)] : Math.floor(Math.random() * 2 ** 31);
  quiz.history = []; quiz.target = quiz.rounds;
  quiz.favorites = []; quiz.more = []; quiz.less = []; quiz.roundExtra = [];
  renderFavs();
  renderQuizFeedback();
  saveQuiz();
  quizShowPlaying();
  $("#notes").replaceChildren();
  $("#results").replaceChildren();
  $("#empty").hidden = false;
  await quizNext();
}

function renderProfile(p) {
  const knew = p.seen ? ` You knew ${p.known} of the ${p.seen} games we showed.` : "";
  $("#quiz-familiarity").textContent = p.picks.length
    ? `${p.familiarity}.${knew}`
    : "Nothing grabbed you yet - try a few more rounds, or describe your dream game." + knew;
  // specific taste: each point with the games it comes from
  const because = (games) => games && games.length ? el("span", { class: "because", text: ` - ${games.join(", ")}` }) : null;
  const fill = (id, items, make) => {
    const ul = $(`#${id}`);
    ul.replaceChildren(...(items || []).map(make));
    $(`#${id}-box`).hidden = !ul.children.length;
  };
  fill("taste-styles", p.styles, (s) => el("li", {}, el("b", { text: s.name }), because(s.because)));
  fill("taste-loves", p.loves, (s) => el("li", {}, el("b", { text: s.label }), because(s.because)));
  fill("taste-moods", p.moods, (m) => el("li", { text: m }));
  fill("taste-praised", p.praised, (s) => el("li", {}, el("b", { text: s.game }),
    el("span", { class: "because", text: ` - ${s.points.join("; ")}` })));
  fill("taste-avoids", p.avoids, (s) => el("li", {}, el("b", { text: s.label }), because(s.because)));
  fill("taste-radar", p.radar, (s) => el("li", {}, el("b", { text: s.label }), because(s.because)));
  const strip = $("#quiz-picks");
  strip.replaceChildren();
  for (const g of p.picks) {
    const classic = IMG(Number(g.appid));
    const img = el("img", { alt: "", loading: "lazy" });
    img.addEventListener("error", () => { if (img.src !== classic) img.src = classic; }, { once: true });
    img.src = safeUrl(g.image, classic);
    strip.append(el("figure", {}, img, el("figcaption", { text: g.name, title: g.name })));
  }
  $("#quiz-picks-label").hidden = !p.picks.length;
  $("#quiz-more").hidden = p.rounds >= quiz.maxRounds;
}

// clear the taste area so a previous quiz's profile never lingers
function resetProfile() {
  $("#quiz-familiarity").textContent = "";
  $("#quiz-picks").replaceChildren();
  $("#quiz-picks-label").hidden = true;
  for (const id of ["styles", "loves", "moods", "praised", "avoids", "radar"]) {
    $(`#taste-${id}`).replaceChildren();
    $(`#taste-${id}-box`).hidden = true;
  }
  quiz.lastProfile = null;
}

async function quizFinish({ focusResults = false } = {}) {
  if (!quiz.history.length) return;
  // quiz results on screen already ("Update results"): recalculate them in place - the
  // old ones stay (dimmed) until the new ones arrive, so the page doesn't jump around
  const update = quizResultsShown();
  const withFeedback = update && pendingChanges() > 0;
  if (!update) { resetProfile(); quiz.resultExtra = null; }
  ++quiz.token;                              // no round may render after this
  quiz.busy = false;
  quiz.current = null;
  const seq = ++state.resultSeq;
  quiz.resultsSeq = seq;
  quiz.done = true;
  saveQuiz();
  for (const id of ["#quiz-board", "#quiz-actions", "#quiz-hint", ".quiz-keys", ".quiz-head", "#quiz-howto"]) $(id).hidden = true;
  $("#quiz-done").hidden = false;
  $("#fav-msg").textContent = "";
  // the update bar hides while this runs: don't leave keyboard focus on it
  if ($("#feedback-bar").contains(document.activeElement)) $("#results-heading").focus({ preventScroll: true });
  setMode("quiz", false);
  state.lastQuery = { mode: "quiz" };
  state.busy = false;
  const extra = quizExtra();
  const go = $("#go");
  go.disabled = true; go.textContent = "Searching…"; go.classList.remove("pending");
  $("#empty").hidden = true;
  $("#results").setAttribute("aria-busy", "true");
  if (update) {
    $("#results").classList.add("updating");
    showNotes(["Updating your games…"]);
    announce("Updating your results…");
  } else {
    showNotes(["Finding your games…"]);
    skeleton();
  }
  saveFilters();
  try {
    const data = await api("/api/quiz/result", { history: quiz.history, options: readOptions(), extra });
    if (seq !== state.resultSeq) return;
    renderProfile(data.profile);
    quiz.lastProfile = data.profile;
    const rows = data.results || [];
    showNotes(data.notes || []);
    quiz.resultExtra = extra;
    renderResults(rows);
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (update) {
      announce(`Results updated${withFeedback ? " with your feedback" : ""}. ${rows.length} games recommended.`);
      // focus was on a card that's gone (or on the bar): to the results
      const a = document.activeElement;
      if (!focusResults && (!a || a === document.body)) $("#results-heading").focus({ preventScroll: true });
      // stay with the results: back to where they start if that's off screen - unless
      // they're typing somewhere (e.g. another favourite) while the update ran
      const typing = a && a.closest && a.closest("input, textarea, select, [contenteditable]");
      const top = $("#notes").getBoundingClientRect().top;
      if (!typing && (top < 0 || top > window.innerHeight * 0.6)) $("#notes").scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
    } else {
      announce(`Quiz done. ${rows.length} games recommended.`);
      if (!focusResults) $("#quiz-done-title").focus({ preventScroll: true });
      const top = $("#quiz-done").getBoundingClientRect().top;
      if (top < 0 || top > window.innerHeight * 0.6) $("#quiz-done").scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
    }
  } catch (e) {
    if (seq !== state.resultSeq) return;
    if (!update) {                           // an update keeps the old results; the bar offers another try
      $("#results").replaceChildren();
      $("#empty").hidden = false;
    }
    showNotes([e.message], "error");
  } finally {
    if (seq === state.resultSeq) {
      quiz.resultsSeq = -1;
      $("#results").classList.remove("updating");
      go.disabled = false;
      $("#results").setAttribute("aria-busy", "false");
      updateFeedbackBar();                   // also sets the #go label
      if (focusResults) $("#results-heading").focus({ preventScroll: true });
    }
  }
}

function validSavedRound(h) {
  if (!h || !Array.isArray(h.shown) || h.shown.length > 5) return false;
  if ("order" in h) {
    // the boxes are optional (older saves only have "unknown"); every placed game
    // was shown and is in one place only
    const boxes = [h.unknown, h.no_interest, h.interested];
    if (!Array.isArray(h.order) || !boxes.every((l) => l == null || Array.isArray(l))) return false;
    const all = [h.order, ...boxes].flatMap((l) => l || []);
    return all.every((a) => h.shown.includes(a)) && new Set(all).size === all.length;
  }
  return h.pick === null || h.shown.includes(h.pick);   // older saved format
}

async function quizResume() {
  const saved = store.get("dgf.quiz", null);
  const ok = saved && Number.isInteger(saved.seed) && Array.isArray(saved.history) &&
    saved.history.length <= quiz.maxRounds && saved.history.every(validSavedRound);
  if (!ok) return quizStart();
  restoreExtra(saved);
  // nothing played yet: a fresh start - unless they'd added favourites during round 1
  if (!saved.history.length && !quiz.favorites.length) return quizStart();
  quiz.seed = saved.seed;
  quiz.history = saved.history;
  // per-round snapshots (older saves have none: those rounds are made with the current extra)
  quiz.roundExtra = Array.isArray(saved.roundExtra) && saved.roundExtra.length <= quiz.maxRounds + 1
    ? saved.roundExtra.map((x) => (validExtraSnapshot(x) ? x : undefined)) : [];
  quiz.target = Math.min(quiz.maxRounds, Math.max(quiz.rounds, Number(saved.target) || quiz.rounds));
  if (quiz.history.length && (saved.done || quiz.history.length >= quiz.target)) {
    const shown = quizFinish();
    fillExtraNames();                        // after the results request, so it doesn't wait behind this one
    return shown;
  }
  saveQuiz();                                // the cleaned-up lists (and an older save's hidden games)
  quizShowPlaying();
  const round = quizNext({ resuming: true });
  fillExtraNames();
  await round;
}

// keys: 1-5 rank / un-rank that card, Shift+1-5 "never heard of it",
// Enter next round, Backspace undo the last placement, 0 "I don't like any of them"
// (N / H / I for the boxes work on a focused card - see renderRound)
function quizKeys(e) {
  if (drag.st) {                               // keys do nothing mid-drag, except Esc = put it back
    if (e.key === "Escape") { e.preventDefault(); dragCancel(); announce("Drag cancelled"); }
    return;
  }
  if (e.defaultPrevented || e.repeat || state.mode !== "quiz" || quiz.done || quiz.busy || quiz.loading ||
      e.ctrlKey || e.metaKey || e.altKey) return;
  const t = e.target;
  if (t.closest && t.closest("input, textarea, select, [contenteditable]")) return;
  const digit = /^Digit([0-5])$/.exec(e.code || "") || /^Numpad([0-5])$/.exec(e.code || "");
  const d = digit ? Number(digit[1]) : /^[0-5]$/.test(e.key) ? Number(e.key) : null;
  if (d !== null && d >= 1) {
    const g = quiz.current?.games[d - 1];
    if (!g) return;
    e.preventDefault();
    if (e.shiftKey) quizToggleBox(g.appid, "unk"); else quizToggleRank(g.appid);
  } else if (d === 0 && !e.shiftKey) {
    e.preventDefault(); quizSubmit(true);
  } else if (e.key === "Enter" && !$("#quiz-next").disabled && (!t.closest("button, a") || t.closest(".qcard"))) {
    e.preventDefault(); quizSubmit(false);
  } else if (e.key === "Backspace" && quiz.undo.length && (!t.closest("button, a") || t.closest(".qcard"))) {
    e.preventDefault(); quizUndo();
  }
}

/* ---------------- drag and drop (mouse, touch and pen) ---------------- */
// Pointer events on the document, so a drag keeps working while the card is
// moved between zones for the live preview. The dragged card itself shows
// where it will land; a floating copy follows the pointer.
const drag = { st: null, raf: 0, afterDrop: null };
const DRAG_START_PX = 6;      // mouse: a drag starts after this much movement
const HOLD_MS = 220;          // touch / pen: press and hold this long, so a swipe still scrolls
const HOLD_SLOP_PX = 10;      // ... moving further than this first means "scroll"
const DRAG_FROZEN = ["#quiz-choices", "#quiz-rank", "#quiz-bins"];   // heights held while a card is up

function dragDown(e) {
  if (drag.st) return;
  const card = e.target.closest?.(".qcard");
  if (!card || !card.closest("#quiz-board") || quiz.busy || quiz.loading || !quiz.current || quiz.done) return;
  if (e.pointerType === "mouse" && e.button !== 0) return;
  const appid = Number(card.dataset.appid);
  if (!quiz.cards.has(appid)) return;
  const touchy = e.pointerType === "touch" || e.pointerType === "pen";
  if (!touchy) e.preventDefault();                      // no text selection / native image drag
  const r = card.getBoundingClientRect();
  const st = drag.st = { appid, card, id: e.pointerId, x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY,
    offX: e.clientX - r.left, offY: e.clientY - r.top, w: r.width, started: false, target: null, key: "?",
    touchy, timer: 0 };
  if (touchy) {
    card.classList.add("pressing");
    st.timer = setTimeout(() => {
      if (drag.st !== st || st.started) return;
      card.classList.remove("pressing");
      navigator.vibrate?.(10);
      dragBegin(st);
      dragRetarget(st);
    }, HOLD_MS);
  }
}

function dragMove(e) {
  const st = drag.st;
  if (!st || e.pointerId !== st.id) return;
  st.x = e.clientX; st.y = e.clientY;
  if (!st.started) {
    const moved = Math.hypot(st.x - st.x0, st.y - st.y0);
    if (st.touchy) {                                    // moved before the hold: the user is scrolling
      if (moved > HOLD_SLOP_PX) { clearTimeout(st.timer); st.card.classList.remove("pressing"); drag.st = null; }
      return;
    }
    if (moved < DRAG_START_PX) return;
    dragBegin(st);
  }
  e.preventDefault();
  placeGhost(st);
  dragRetarget(st);
}

function dragBegin(st) {
  st.started = true;
  // freeze zone heights so the preview can't shift the targets under the pointer
  // (the boxes may still grow, but never shrink away from it)
  for (const id of DRAG_FROZEN) {
    const z = $(id);
    z.style.minHeight = `${z.getBoundingClientRect().height}px`;
  }
  const ghost = st.card.cloneNode(true);
  ghost.classList.add("drag-ghost");
  ghost.classList.remove("dragging");
  ghost.setAttribute("aria-hidden", "true");
  ghost.style.width = `${st.w}px`;
  document.body.append(ghost);
  st.ghost = ghost;
  document.body.classList.add("is-dragging");
  st.card.classList.add("dragging");
  placeGhost(st);
  drag.raf = requestAnimationFrame(autoScroll);
}

function placeGhost(st) {
  st.ghost.style.transform = `translate(${st.x - st.offX}px, ${st.y - st.offY}px) rotate(1.5deg)`;
}

// index of the element whose centre is nearest to (x, y)
function nearestIndex(nodes, x, y) {
  let best = 0, bestD = Infinity;
  nodes.forEach((s, i) => {
    const r = s.getBoundingClientRect();
    const d = Math.hypot(x - (r.left + r.width / 2), y - (r.top + r.height / 2));
    if (d < bestD) { bestD = d; best = i; }
  });
  return best;
}

// which zone (and, in the ranking, which place) is under the pointer
function dropTargetAt(x, y) {
  const hit = document.elementFromPoint(x, y);
  let zone = hit?.closest?.("#quiz-board .dz");
  const bins = !zone && hit?.closest?.("#quiz-bins");
  if (bins) {                                             // in the gap between two boxes: the nearer one
    const boxes = bins.querySelectorAll(".dz");
    zone = boxes[nearestIndex(boxes, x, y)];
  }
  if (!zone) return null;
  if (zone.dataset.zone !== "rank") return { zone: zone.dataset.zone, index: -1 };
  return { zone: "rank", index: nearestIndex(zone.querySelectorAll(".slot"), x, y) };
}

// the round as it would be if the card were dropped on target t; the card
// leaves every other zone
function previewFor(appid, t) {
  if (!t) return placements();
  const p = withoutGame(quiz, appid);
  if (t.zone === "rank") p.order.splice(Math.min(t.index, p.order.length), 0, appid);
  else if (BOX[t.zone]) {
    const k = BOX[t.zone].key;
    p[k] = quiz[k].includes(appid) ? [...quiz[k]] : [...p[k], appid];   // already in that box: stays put
  }
  return p;
}

function dragRetarget(st) {
  const t = dropTargetAt(st.x, st.y);
  const key = t ? `${t.zone}:${t.index}` : "";
  if (key === st.key) return;
  st.key = key;
  st.target = t;
  // a card heading back to the deck stays put until it's dropped, so the deck
  // never grows under the pointer
  layoutRound(t && t.zone !== "deck" ? previewFor(st.appid, t) : quiz, st.appid);
  document.querySelectorAll("#quiz-board .dz").forEach((z) => z.classList.toggle("over", !!t && z.dataset.zone === t.zone));
}

function dragUp(e) {
  const st = drag.st;
  if (!st || e.pointerId !== st.id) return;
  if (!st.started) {                                    // a plain click / tap: games are sorted by dragging
    clearTimeout(st.timer);
    st.card.classList.remove("pressing");
    drag.st = null;
    const name = gameName(st.appid);
    $("#quiz-status").textContent = st.touchy ? `Press and hold “${name}”, then drag it into your ranking or a box below.`
      : `Drag “${name}” into your ranking or a box below (or press its number key).`;
    const rank = $("#quiz-rank");
    rank.classList.remove("nudge"); void rank.offsetWidth; rank.classList.add("nudge");
    announce($("#quiz-status").textContent);
    return;
  }
  st.x = e.clientX; st.y = e.clientY;
  dragRetarget(st);
  dragEnd(st, st.target);
}

function dragCancel() {
  if (drag.st) dragEnd(drag.st, null);
}

function dragEnd(st, t) {
  cancelAnimationFrame(drag.raf);
  clearTimeout(st.timer);
  st.card.classList.remove("pressing");
  st.ghost?.remove();
  drag.st = null;
  document.body.classList.remove("is-dragging");
  for (const id of DRAG_FROZEN) $(id).style.minHeight = "";
  document.querySelectorAll("#quiz-board .dz.over").forEach((z) => z.classList.remove("over"));
  if (drag.afterDrop) { const f = drag.afterDrop; drag.afterDrop = null; f(); }
  if (!st.started || !quiz.cards.has(st.appid) || !quiz.current) return;
  if (t) commitPlacements(previewFor(st.appid, t), st.appid);
  updateRoundUI();
  if (t) {
    announce(`${gameName(st.appid)}: ${whereIs(st.appid)}. ${$("#quiz-status").textContent}`);
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!reduce) { st.card.classList.remove("dropped"); void st.card.offsetWidth; st.card.classList.add("dropped"); }
  }
}

// scroll the page while a card is held near the top or bottom edge
function autoScroll() {
  const st = drag.st;
  if (!st || !st.started) return;
  const edge = 70, h = window.innerHeight;
  const dy = st.y < edge ? -Math.ceil((edge - st.y) / 5) : st.y > h - edge ? Math.ceil((st.y - (h - edge)) / 5) : 0;
  if (dy) {
    const before = window.scrollY;
    window.scrollBy(0, dy);
    if (window.scrollY !== before) { st.key = "?"; dragRetarget(st); }
  }
  drag.raf = requestAnimationFrame(autoScroll);
}

function dragSetup() {
  document.addEventListener("pointerdown", dragDown);
  document.addEventListener("pointermove", dragMove, { passive: false });
  document.addEventListener("pointerup", dragUp);
  document.addEventListener("pointercancel", (e) => { if (drag.st && e.pointerId === drag.st.id) dragCancel(); });
  // once a touch drag has started, the page must not scroll under the finger
  document.addEventListener("touchmove", (e) => { if (drag.st?.started) e.preventDefault(); }, { passive: false });
  document.addEventListener("contextmenu", (e) => { if (drag.st) e.preventDefault(); });
  if (window.matchMedia("(pointer: coarse)").matches) {
    $("#quiz-howto").replaceChildren(
      el("b", { text: "Press and hold a game you've played, then drag it into your ranking, favourite first." }),
      " Haven't played one? Drop it in the box that fits: ", el("i", { text: "Heard of it, no interest" }), ", ",
      el("i", { text: "Never heard of it" }), " or ", el("i", { text: "Interested, haven't played" }), ".");
  }
  window.addEventListener("blur", dragCancel);
  $("#quiz-board").addEventListener("dragstart", (e) => e.preventDefault());
}

/* ---------------- init ---------------- */
async function init() {
  // tabs (roving tabindex + arrow keys)
  for (const m of MODES) $(`#tab-${m}`).addEventListener("click", () => openTab(m));
  $(".tabs").addEventListener("keydown", (e) => {
    if (["ArrowRight", "ArrowLeft", "Home", "End"].includes(e.key)) {
      e.preventDefault();
      const i = MODES.indexOf(state.mode);
      const next = e.key === "Home" ? MODES[0] : e.key === "End" ? MODES[MODES.length - 1]
        : MODES[(i + (e.key === "ArrowRight" ? 1 : MODES.length - 1)) % MODES.length];
      openTab(next, false);
      $(`#tab-${next}`).focus();
    }
  });

  // quiz
  buildRankSlots();
  dragSetup();
  $("#quiz-none").addEventListener("click", () => quizSubmit(true));
  $("#quiz-next").addEventListener("click", () => quizSubmit(false));
  $("#quiz-clear").addEventListener("click", quizClearRound);
  $("#quiz-back").addEventListener("click", quizBack);
  $("#quiz-restart").addEventListener("click", quizStart);
  $("#quiz-again").addEventListener("click", () => { quizStart().then(() => $("#quiz-title").focus()); });
  $("#quiz-finish").addEventListener("click", () => {
    // a fully sorted current round still counts
    const rd = quiz.current;
    if (rd && !quiz.busy && placedCount() === rd.games.length) {
      quiz.history.push(roundEntry(rd.games.map((g) => g.appid), false));
      saveQuiz();
    }
    quizFinish();
  });
  $("#quiz-more").addEventListener("click", async () => {
    cancelResults();                         // ignore a results request still in flight
    quiz.target = Math.min(quiz.maxRounds, quiz.history.length + 3);
    saveQuiz();
    quizShowPlaying();
    await quizNext();
    $("#quiz-title").focus();
  });
  $("#quiz-tune").addEventListener("click", () => {
    const favs = quiz.lastProfile?.picks || [];
    state.liked = favs.slice(0, MAX_GAMES).map((g) => ({ appid: g.appid, name: g.name, neg: false }));
    renderLiked();
    setMode("like");
  });
  document.addEventListener("keydown", quizKeys);

  // picker
  const input = $("#game-input");
  gameCombo({
    input, list: $("#suggest"), prefix: "sg",
    skip: (g) => state.liked.some((x) => x.appid === g.appid),
    onPick: (g) => addGame(g),
    onEmptyEnter: () => { if (state.liked.length) run(); },
    onEmptyBackspace: () => { if (state.liked.length) { state.liked.pop(); renderLiked({ input: true }); } },
  });
  $("#picker").addEventListener("click", (e) => { if (e.target === e.currentTarget) input.focus(); });

  // quiz: favourites they add themselves
  const favInput = $("#fav-input");
  gameCombo({
    input: favInput, list: $("#fav-suggest"), prefix: "fsg",
    skip: (g) => quiz.favorites.some((x) => x.appid === g.appid),
    note: (g) => (inCurrentRound(g.appid) ? "in this round" : null),
    onPick: addFavorite,
  });
  favInput.addEventListener("input", () => { $("#fav-msg").textContent = ""; });
  $("#fav-picker").addEventListener("click", (e) => { if (e.target === e.currentTarget) favInput.focus(); });
  $("#feedback-update").addEventListener("click", () => run());

  // dream
  const ex = $("#examples");
  for (const t of EXAMPLES) ex.append(el("button", { class: "example", type: "button", text: t,
    onclick: () => { $("#dream-text").value = t; updateGoogleLink(); $("#dream-text").focus(); } }));
  $("#dream-text").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); askGoogle(); } });
  $("#dream-text").addEventListener("input", updateGoogleLink);
  $("#google-ai").addEventListener("click", (e) => {
    e.preventDefault();                         // window.open keeps the hash saved and the wish checked
    askGoogle();
  });
  updateGoogleLink();
  if (!publicSite()) $(".google-note").textContent = "Opens Google's AI Mode in a new tab with your wish. Free, no sign-in needed.";

  // filters
  tagInput("#must-input", "must");
  tagInput("#excl-input", "excl");
  for (const id of ["#f-gems", "#f-free", "#f-platform", "#f-positive", "#f-year"]) $(id).addEventListener("change", saveFilters);
  $("#go").addEventListener("click", () => run());
  loadFilters();
  setMode("quiz", false);

  try {
    state.status = await api("/api/status");
  } catch (e) {
    $("#stats").textContent = "server not reachable";
    showNotes([e.message], "error");
    return;
  }
  const s = state.status;
  $("#stats").textContent = `${fmt(s.games)} games · ${fmt(s.players)} players' reviews`;
  const dl = $("#tag-list");
  for (const t of s.tags) dl.append(el("option", { value: t }));
  const st = $("#starters");
  for (const g of s.starters.slice(0, 14)) st.append(el("button", { class: "starter", type: "button", text: g.name, onclick: () => addGame(g) }));
  const ai = $("#ai-toggle");
  $("#ai-row").hidden = !s.ai.available;
  if (!s.ai.available) {
    ai.disabled = true;
    ai.setAttribute("aria-describedby", "ai-reason");
    $("#ai-reason").textContent = `AI mode is off: ${s.ai.reason}.`;
  } else {
    ai.checked = store.get("dgf.ai", false);
  }
  ai.addEventListener("change", () => store.set("dgf.ai", ai.checked));

  if (s.quiz) { quiz.rounds = s.quiz.rounds; quiz.maxRounds = s.quiz.max_rounds; }
  const fromLink = await loadHash();
  if (fromLink === "run") run();
  else if (!fromLink && !quiz.current && !quiz.loading) await quizResume();   // otherwise it starts when the quiz tab is opened
  // a pasted/edited link while the page is open (replaceState doesn't fire this)
  window.addEventListener("hashchange", async () => {
    if ((await loadHash()) === "run") run();
  });
}

init();
