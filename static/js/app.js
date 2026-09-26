/* ==========================================================================
   app.js (front end): epicycle engine, transitions and UI.
   The vision and the Fourier maths run in the Python backend (app.py); this file
   uploads the photo, asks the server for contours and coefficients, and animates them.
   photo → [Python: face, level, crop, features, contours, FFT] → [browser: morph + epicycle animation]
   ========================================================================== */
(function () {
'use strict';

/* ---------- epicycle constants and helpers (must match epicycle/fourier.py) ---------- */
const G = 8192, GM = G - 1, F = 1024, NF = 2 * F + 1;
const COS = new Float64Array(G), SIN = new Float64Array(G);
for (let i = 0; i < G; i++) { const a = 2 * Math.PI * i / G; COS[i] = Math.cos(a); SIN[i] = Math.sin(a); }

/* tip of the epicycle chain at sample j using the n biggest circles listed in `order` (table lookup, no trig) */
function evalAt(re, im, order, n, j, out) {
  let x = re[F], y = im[F];
  for (let k = 0; k < n; k++) {
    const i = order[k], m = ((i - F) * j) & GM, c = COS[m], s = SIN[m];
    x += re[i] * c - im[i] * s;
    y += re[i] * s + im[i] * c;
  }
  out[0] = x; out[1] = y;
}

/* replay the user's edits on top of the automatic strokes.
   erase: { type, rects, whole } removes what is under the squares.
   add:   { type, items: [{ rect, strokes, parts }] } each square replaces what was there with fresh contours. */
function applyEdits(strokes, parts, edits) {
  let S = strokes.slice(), P = parts.slice();
  const cut = (rects, whole) => {
    const inR = (x, y) => { for (let r = 0; r < rects.length; r++) { const q = rects[r]; if (x >= q[0] && x <= q[2] && y >= q[1] && y <= q[3]) return true; } return false; };
    const nS = [], nP = [];
    for (let k = 0; k < S.length; k++) {
      const s = S[k];
      if (whole) {
        let hit = false;
        for (let i = 0; i < s.length && !hit; i += 2) hit = inR(s[i], s[i + 1]);
        if (!hit) { nS.push(s); nP.push(P[k]); }
        continue;
      }
      let run = [];
      const flush = () => { if (run.length >= 8) { nS.push(run.length === s.length ? s : Float32Array.from(run)); nP.push(P[k]); } run = []; };
      for (let i = 0; i < s.length; i += 2) { if (inR(s[i], s[i + 1])) flush(); else run.push(s[i], s[i + 1]); }
      flush();
    }
    S = nS; P = nP;
  };
  for (const e of edits) {
    if (e.type === 'erase') cut(e.rects, e.whole);
    else for (const it of e.items) { cut([it.rect], false); for (let k = 0; k < it.strokes.length; k++) { S.push(it.strokes[k]); P.push(it.parts[k]); } }
  }
  return { strokes: S, parts: P };
}
const Core = { evalAt, applyEdits };
const KMAX = NF - 1;                       // every harmonic except the centre (DC) term
const TAU = Math.PI * 2;
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const reduceMotion = !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);
const T = (ms) => (reduceMotion ? ms * 0.35 : ms);
const $ = (id) => document.getElementById(id);

/* ---------- facial parts ---------- */
const PARTS = [null,
  { key: 'hair', name: 'Hair' }, { key: 'face', name: 'Face' }, { key: 'eyes', name: 'Eyes' }, { key: 'nose', name: 'Nose' },
  { key: 'lips', name: 'Lips' }, { key: 'beard', name: 'Beard' }, { key: 'ears', name: 'Ears' }, { key: 'body', name: 'Neck & shoulders' }];
const CHIP_ORDER = [1, 3, 4, 5, 6, 7, 2, 8];
const DRAW_ORDER = [1, 7, 2, 6, 3, 4, 5, 8];          // hair, ears, face, beard, eyes, nose, lips, body
const partOn = [false, true, true, true, true, true, true, true, true];
const partPresent = [false, false, false, false, false, false, false, false, false];

/* ---------- DOM ---------- */
const stage = $('stage');
const cRef = $('cRef'), cStr = $('cStr'), cTrail = $('cTrail'), cLive = $('cLive');
const xRef = cRef.getContext('2d'), xStr = cStr.getContext('2d');
const xTrail = cTrail.getContext('2d'), xLive = cLive.getContext('2d');
const layers = [cRef, cStr, cTrail, cLive];
const elStatus = $('status'), elReadout = $('readout'), elBar = $('bar');

/* ---------- state ---------- */
let W = 0, H = 0, DPR = 1, lw = 1.2;
const view = { ox: 0, oy: 0, s: 1 };
const col = { ink: '#10294a', accent: '#c47600', paper: '#edf3f5', danger: '#a3241c', part: [] };

const params = { detail: 0.72, sigma: 1.4, minLen: 10 };
let curM = 199, curK = 398, drawSecs = 24;      // curM = highest harmonic used, curK = 2*curM circles
let nOrd = 0, kTouched = false;
let showCircles = true, showArms = true, loopOn = false, colorParts = true, refMode = 'none';
/* contour editing: 'focus' finds new contours under a square, 'erase' removes them */
let tool = null, sizePx = 56, focusLevel = 0.6, eraseWhole = false;
let edits = [], gesture = null, cursor = null, lastApplied = null, rebuildTimer = 0;

/* the coefficients currently on screen (blended in place while morphing) */
const disp = { re: new Float64Array(NF), im: new Float64Array(NF), amp: new Float64Array(NF) };
/* what is actually drawn: harmonics up to curM, softly tapered (Lanczos) so the line stays smooth */
const eff = { re: new Float64Array(NF), im: new Float64Array(NF), amp: new Float64Array(NF) };
let order = new Int32Array(KMAX);
let curPen = new Uint8Array(G), curPart = new Uint8Array(G);
disp.re[F + 1] = 0.42;                     // start life as one big circle

function applyTaper() {
  const m = curM;
  for (let i = 0; i < NF; i++) {
    const f = i - F;
    if (f === 0) { eff.re[i] = disp.re[i]; eff.im[i] = disp.im[i]; eff.amp[i] = 0; continue; }
    if (f > m || f < -m) { eff.re[i] = 0; eff.im[i] = 0; eff.amp[i] = 0; continue; }
    const x = Math.PI * f / (m + 1), sn = Math.sin(x) / x;
    eff.re[i] = disp.re[i] * sn; eff.im[i] = disp.im[i] * sn;
    eff.amp[i] = Math.hypot(eff.re[i], eff.im[i]);
  }
}
function buildOrder() {                    // biggest circle first
  const idx = [];
  for (let f = -curM; f <= curM; f++) if (f) idx.push(f + F);
  idx.sort((a, b) => eff.amp[b] - eff.amp[a]);
  order = Int32Array.from(idx); nOrd = idx.length;
}
applyTaper(); buildOrder();

const trail = { x: new Float32Array(G + 4), y: new Float32Array(G + 4), j: new Int32Array(G + 4), n: 0, last: -1 };
let src = null;        // { kind: sample|photo|plain, w, h, rgba, base, strokes, parts, analysis, refCanvas }
let lastBuilt = null, boxes = [];
let morph = null;
let phase = 'morph';                       // morph | draw | done | fade
let jf = 0, playing = true, chainA = 1, doneAt = 0, fadeAt = 0;
let annoA = 0, annoTarget = 0, annoPupils = 0;
let userStarted = false, seq = 0, needRebuild = false, lastT = performance.now(), lastPct = -1;
const tmp = [0, 0];
const chainX = new Float32Array(KMAX + 2), chainY = new Float32Array(KMAX + 2);

/* ---------- small helpers ---------- */
const sleep = (ms, token) => new Promise((r) => setTimeout(() => r(token === seq), ms));
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
const fmt = (n) => n.toLocaleString('en-US');

function setStatus(msg, isErr) {
  elStatus.textContent = msg;
  elStatus.classList.toggle('err', !!isErr);
}
const STEP_ORDER = ['src', 'features', 'fourier', 'draw'];
function setStep(name) {
  const idx = STEP_ORDER.indexOf(name);
  document.querySelectorAll('#steps li').forEach((li) => {
    const i = STEP_ORDER.indexOf(li.dataset.s);
    li.classList.toggle('done', i < idx);
    li.classList.toggle('active', i === idx);
  });
}
function readColors() {
  const cs = getComputedStyle(document.documentElement);
  const g = (n, d) => cs.getPropertyValue(n).trim() || d;
  col.ink = g('--ink', col.ink); col.accent = g('--accent', col.accent); col.paper = g('--paper', col.paper); col.danger = g('--danger', col.danger);
  for (let p = 1; p < PARTS.length; p++) col.part[p] = g('--p' + p, col.ink);
  col.font = getComputedStyle(document.body).fontFamily || 'sans-serif';
}
const partColor = (p) => (colorParts && p ? col.part[p] || col.ink : col.ink);
const autoM = (len) => clamp(Math.round(len * 0.15), 150, F);       // enough harmonics to resolve eyes and lips
function setCircleCount(m) {               // update state + slider without triggering a rebuild
  curM = clamp(m, 2, F); curK = 2 * curM;
  const el = $('rK'); el.value = Math.round(Math.log(curM / 2) / Math.log(F / 2) * 100);
  paintSlider(el); $('oK').textContent = fmt(curK);
}
const mFromSlider = (v) => Math.round(2 * Math.pow(F / 2, v / 100));   // 2 … F harmonics each way

/* ---------- sizing and static painting ---------- */
function resize() {
  const r = stage.getBoundingClientRect();
  const w = Math.round(r.width), h = Math.round(r.height);
  if (w < 60 || h < 60) return;
  W = w; H = h; DPR = Math.min(window.devicePixelRatio || 1, 2.5);
  layers.forEach((c) => { c.width = Math.round(W * DPR); c.height = Math.round(H * DPR); });
  const top = W < 700 ? 44 : 60, bottom = W < 700 ? 40 : 48;
  const availH = Math.max(120, H - top - bottom), availW = W - 20;
  view.s = Math.min(availW, availH) / 2;
  view.ox = W / 2; view.oy = top + availH / 2;
  lw = clamp(view.s / 360, 1.05, 2);
  repaintStatic();
}

function screenScale() { return src ? view.s / (Math.max(src.w, src.h) / 2) : 1; }
const toScreen = (x, y) => { const k = screenScale(); return [view.ox + (x + 0.5 - src.w / 2) * k, view.oy + (y + 0.5 - src.h / 2) * k]; };

function paintImage(ctx, img) {
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0); ctx.clearRect(0, 0, W, H);
  if (!src || !img) return;
  const k = screenScale(), dw = src.w * k, dh = src.h * k;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, view.ox - dw / 2, view.oy - dh / 2, dw, dh);
}
function visible() {                       // strokes of the parts that are switched on
  const strokes = [], parts = [];
  if (!src || !src.strokes) return { strokes, parts };
  for (let i = 0; i < src.strokes.length; i++) {
    const p = src.parts[i];
    if (p === 0 || partOn[p]) { strokes.push(src.strokes[i]); parts.push(p); }
  }
  return { strokes, parts };
}
function drawStrokeSet(c) {
  if (!src || !src.strokes) return;
  const k = screenScale();
  c.lineWidth = Math.max(1, lw * 0.95); c.lineJoin = 'round'; c.lineCap = 'round';
  for (let p = 0; p < PARTS.length; p++) {
    if (p && !partOn[p]) continue;
    c.beginPath(); let any = false;
    for (let n = 0; n < src.strokes.length; n++) {
      if (src.parts[n] !== p) continue;
      const st = src.strokes[n]; any = true;
      for (let i = 0; i < st.length; i += 2) {
        const x = view.ox + (st[i] + 0.5 - src.w / 2) * k, y = view.oy + (st[i + 1] + 0.5 - src.h / 2) * k;
        if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
      }
    }
    if (any) { c.strokeStyle = partColor(p); c.stroke(); }
  }
}
function paintStrokes() {
  const c = xStr;
  c.setTransform(DPR, 0, 0, DPR, 0, 0); c.clearRect(0, 0, W, H);
  drawStrokeSet(c);
}
function drawTrailRange(n0, n1) {
  const c = xTrail, s = view.s, ox = view.ox, oy = view.oy;
  c.setTransform(DPR, 0, 0, DPR, 0, 0);
  c.lineWidth = lw; c.lineJoin = 'round'; c.lineCap = 'round';
  let down = false, cur = null;
  c.beginPath();
  for (let k = Math.max(n0, 1); k < n1; k++) {
    const jp = trail.j[k - 1] % G;
    if (curPen[jp]) { down = false; continue; }                 // pen lifted: no ink
    const colr = partColor(curPart[jp]);
    if (colr !== cur) { if (cur !== null) c.stroke(); c.beginPath(); c.strokeStyle = colr; cur = colr; down = false; }
    const x0 = ox + trail.x[k - 1] * s, y0 = oy + trail.y[k - 1] * s;
    if (!down) { c.moveTo(x0, y0); down = true; }
    c.lineTo(ox + trail.x[k] * s, oy + trail.y[k] * s);
  }
  if (cur !== null) c.stroke();
}
function clearTrailCanvas() { xTrail.setTransform(DPR, 0, 0, DPR, 0, 0); xTrail.clearRect(0, 0, W, H); }
function paintTrailAll() { clearTrailCanvas(); if (trail.n > 1) drawTrailRange(0, trail.n); }
function repaintStatic() {
  paintImage(xRef, src && src.refCanvas);
  paintStrokes(); paintTrailAll();
}

/* ---------- trail (the ink the pen has left) ---------- */
function pushTrail(j, p) {
  const n = trail.n++;
  trail.x[n] = p[0]; trail.y[n] = p[1]; trail.j[n] = j; trail.last = j;
}
function appendTrail(jt) {
  const n0 = trail.n;
  if (trail.last < 0) { Core.evalAt(eff.re, eff.im, order, nOrd, 0, tmp); pushTrail(0, tmp); }
  while (trail.last < jt) {
    const j = trail.last + 1;
    Core.evalAt(eff.re, eff.im, order, nOrd, j, tmp);
    pushTrail(j, tmp);
  }
  if (trail.n > n0) drawTrailRange(n0, trail.n);
}
/* recompute the ink for the current circle count (used when the slider moves) */
function rebuildTrail() {
  const upTo = phase === 'draw' ? Math.floor(jf) : (phase === 'done' || phase === 'fade') ? G : -1;
  trail.n = 0; trail.last = -1;
  if (upTo >= 0) {
    for (let j = 0; j <= upTo; j += 2) { Core.evalAt(eff.re, eff.im, order, nOrd, j, tmp); pushTrail(j, tmp); }
  }
  paintTrailAll();
}

/* ---------- live layer: circles, arms, pen, annotations ---------- */
function drawGhost() {
  const c = xLive, s = view.s;
  c.beginPath();
  for (let j = 0; j <= G; j += 16) {
    Core.evalAt(eff.re, eff.im, order, nOrd, j, tmp);
    const x = view.ox + tmp[0] * s, y = view.oy + tmp[1] * s;
    if (j === 0) c.moveTo(x, y); else c.lineTo(x, y);
  }
  c.strokeStyle = col.ink; c.globalAlpha = 0.22; c.lineWidth = lw; c.lineJoin = 'round'; c.stroke(); c.globalAlpha = 1;
}
function drawChain(tau, alpha) {
  const c = xLive, s = view.s, ox = view.ox, oy = view.oy;
  const re = eff.re, im = eff.im, amp = eff.amp;
  let x = re[F], y = im[F];
  chainX[0] = x; chainY[0] = y;
  const drawC = showCircles && alpha > 0.01;
  if (drawC) c.beginPath();
  for (let n = 0; n < nOrd; n++) {
    const i = order[n], a = TAU * (i - F) * tau, cs = Math.cos(a), sn = Math.sin(a);
    if (drawC) {
      const r = amp[i] * s;
      if (r > 1.6) { const px = ox + x * s, py = oy + y * s; c.moveTo(px + r, py); c.arc(px, py, r, 0, TAU); }
    }
    x += re[i] * cs - im[i] * sn; y += re[i] * sn + im[i] * cs;
    chainX[n + 1] = x; chainY[n + 1] = y;
  }
  if (drawC) { c.strokeStyle = col.accent; c.globalAlpha = 0.3 * alpha; c.lineWidth = 1; c.stroke(); }
  if (showArms && alpha > 0.01) {
    c.beginPath(); c.moveTo(ox + chainX[0] * s, oy + chainY[0] * s);
    for (let n = 1; n <= nOrd; n++) c.lineTo(ox + chainX[n] * s, oy + chainY[n] * s);
    c.strokeStyle = col.accent; c.globalAlpha = 0.8 * alpha; c.lineWidth = 1.1; c.lineJoin = 'round'; c.stroke();
  }
  c.globalAlpha = 1;
  return [x, y];
}
function drawTail() {                                     // the freshest ink glows, then cools into the drawing
  const c = xLive, s = view.s, n = trail.n;
  if (n < 3) return;
  const M = Math.min(n - 1, 240), chunks = 6;
  c.lineWidth = lw * 1.5; c.lineCap = 'round'; c.lineJoin = 'round'; c.strokeStyle = col.accent;
  for (let q = 0; q < chunks; q++) {
    const k0 = n - 1 - M + Math.floor(q * M / chunks), k1 = n - 1 - M + Math.floor((q + 1) * M / chunks);
    c.beginPath(); let down = false;
    for (let k = Math.max(1, k0 + 1); k <= k1; k++) {
      if (curPen[trail.j[k - 1] % G]) { down = false; continue; }
      if (!down) { c.moveTo(view.ox + trail.x[k - 1] * s, view.oy + trail.y[k - 1] * s); down = true; }
      c.lineTo(view.ox + trail.x[k] * s, view.oy + trail.y[k] * s);
    }
    c.globalAlpha = 0.15 + 0.8 * (q + 1) / chunks; c.stroke();
  }
  c.globalAlpha = 1;
}
function drawPen(x, y, alpha) {
  const c = xLive, px = view.ox + x * view.s, py = view.oy + y * view.s;
  const up = curPen[Math.floor(jf) % G] && phase !== 'morph';
  c.globalAlpha = 0.22 * alpha; c.fillStyle = col.accent;
  c.beginPath(); c.arc(px, py, 9, 0, TAU); c.fill();
  c.globalAlpha = alpha; c.beginPath(); c.arc(px, py, 3.6, 0, TAU);
  if (up) { c.strokeStyle = col.accent; c.lineWidth = 1.6; c.stroke(); } else c.fill();
  c.globalAlpha = 1;
}
/* "what the analysis found": corner brackets and names around each feature, plus the pupil line */
function drawAnnotations() {
  if (!src) return;
  const c = xLive;
  if (annoPupils > 0.02 && src.analysis) {
    const a = src.analysis, [x1, y1] = toScreen(a.pupL.x, a.pupL.y), [x2, y2] = toScreen(a.pupR.x, a.pupR.y);
    c.globalAlpha = annoPupils; c.strokeStyle = col.accent; c.fillStyle = col.accent; c.lineWidth = 1.4;
    c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
    for (const [x, y] of [[x1, y1], [x2, y2]]) { c.beginPath(); c.arc(x, y, 4, 0, TAU); c.fill(); c.beginPath(); c.arc(x, y, 11, 0, TAU); c.stroke(); }
    c.font = `500 12px ${col.font}`; c.fillStyle = col.ink; c.textAlign = 'center';
    c.fillText('eyes found, face levelled', (x1 + x2) / 2, Math.min(y1, y2) - 20);
    c.globalAlpha = 1;
  }
  if (annoA < 0.02) return;
  c.font = `500 12px ${col.font}`; c.textAlign = 'left'; c.lineWidth = 1.4; c.lineCap = 'square';
  for (const b of boxes) {
    const [x0, y0] = toScreen(b.x0, b.y0), [x1, y1] = toScreen(b.x1, b.y1), L = Math.min(12, (x1 - x0) / 3, (y1 - y0) / 3);
    c.globalAlpha = annoA; c.strokeStyle = col.part[b.part] || col.ink;
    c.beginPath();
    c.moveTo(x0, y0 + L); c.lineTo(x0, y0); c.lineTo(x0 + L, y0);
    c.moveTo(x1 - L, y0); c.lineTo(x1, y0); c.lineTo(x1, y0 + L);
    c.moveTo(x1, y1 - L); c.lineTo(x1, y1); c.lineTo(x1 - L, y1);
    c.moveTo(x0 + L, y1); c.lineTo(x0, y1); c.lineTo(x0, y1 - L);
    c.stroke();
    if (b.label) { c.fillStyle = col.ink; c.fillText(b.label, x0 + 1, y0 - 6); }
  }
  c.globalAlpha = 1;
}
/* bounding boxes of each visible part, taken from its strokes */
function computeBoxes() {
  boxes = [];
  if (!src || !src.strokes) return;
  const acc = {};
  for (let n = 0; n < src.strokes.length; n++) {
    const p = src.parts[n]; if (!p || !partOn[p]) continue;
    const s = src.strokes[n];
    for (let i = 0; i < s.length; i += 2) {
      let key = String(p);
      if (p === 3 || p === 7) key += s[i] < src.w / 2 ? 'l' : 'r';
      const b = acc[key] || (acc[key] = { part: p, x0: 1e9, y0: 1e9, x1: -1e9, y1: -1e9, left: key.slice(-1) === 'l' });
      if (s[i] < b.x0) b.x0 = s[i]; if (s[i] > b.x1) b.x1 = s[i];
      if (s[i + 1] < b.y0) b.y0 = s[i + 1]; if (s[i + 1] > b.y1) b.y1 = s[i + 1];
    }
  }
  const pad = 6;
  for (const k in acc) {
    const b = acc[k], showLabel = !(b.part === 3 || b.part === 7) || b.left;
    if (b.part === 8 || b.part === 2 || b.x1 - b.x0 < 8) continue;          // body / face outline: not boxed
    boxes.push({ part: b.part, x0: b.x0 - pad, y0: b.y0 - pad, x1: b.x1 + pad, y1: b.y1 + pad, label: showLabel ? PARTS[b.part].name.toLowerCase() : '' });
  }
}

/* ---------- morphing between two sets of circles ---------- */
function startMorph(built, dur) {
  const fromRe = Float64Array.from(disp.re), fromIm = Float64Array.from(disp.im);
  const a = new Float64Array(NF), idx = [];
  for (let i = 0; i < NF; i++) {
    a[i] = Math.max(Math.hypot(fromRe[i], fromIm[i]), Math.hypot(built.re[i], built.im[i]));
    if (i !== F) idx.push(i);
  }
  idx.sort((p, q) => a[q] - a[p]);
  morph = { fromRe, fromIm, toRe: built.re, toIm: built.im, pen: built.pen, part: built.part, order: Int32Array.from(idx), t0: performance.now(), dur };
  order = morph.order; nOrd = KMAX;
  phase = 'morph';
  cTrail.style.opacity = 0;                               // old drawing melts away while the circles re-tune
  needRebuild = false;
}
function stepMorph(now) {
  const p = clamp((now - morph.t0) / morph.dur, 0, 1), e = ease(p);
  for (let i = 0; i < NF; i++) {
    disp.re[i] = morph.fromRe[i] + (morph.toRe[i] - morph.fromRe[i]) * e;
    disp.im[i] = morph.fromIm[i] + (morph.toIm[i] - morph.fromIm[i]) * e;
  }
  applyTaper();
  if (p >= 1) finishMorph();
}
function finishMorph() {
  if (!morph) return;
  disp.re.set(morph.toRe); disp.im.set(morph.toIm);
  curPen = morph.pen; curPart = morph.part;
  applyTaper(); buildOrder();
  morph = null;
  startDrawing();
}
const trailOpacity = () => (tool ? 0.22 : 1);
function startDrawing() {
  jf = 0; trail.n = 0; trail.last = -1; lastPct = -1;
  clearTrailCanvas();
  cTrail.style.opacity = trailOpacity();
  phase = 'draw';
  setStep('draw');
  setStatus('Drawing');
}
function finishNow() {
  if (morph) finishMorph();
  if (phase === 'done') return;
  phase = 'done'; doneAt = performance.now(); jf = 0;
  rebuildTrail();
  cTrail.style.opacity = trailOpacity();
  setStatus('Finished. Replay it, or change the settings.');
  setStep('draw');
}

/* ---------- reference layers, readouts, feature chips ---------- */
function applyRefMode() {
  const editing = !!tool && !!src;
  cRef.style.opacity = src && src.refCanvas && (editing || refMode === 'photo') ? (editing ? 0.38 : 0.3) : 0;
  cStr.style.opacity = editing ? 0.95 : refMode === 'contours' ? 0.5 : 0;
  if (!morph && phase !== 'fade') cTrail.style.opacity = trailOpacity();
}
function updateReadout() {
  const st = lastBuilt ? lastBuilt.stats : null;
  elReadout.textContent = `${fmt(curK)} of ${fmt(KMAX)} circles` + (st ? ` · ${fmt(st.strokes)} strokes` : '');
}
function updateControls() {
  const kind = src ? src.kind : 'sample';
  const photoish = kind === 'photo' || kind === 'plain';
  $('fsContours').disabled = !photoish;
  $('rdPhoto').disabled = !(src && src.refCanvas);
  if (refMode === 'photo' && $('rdPhoto').disabled) { document.querySelector('input[name=ref][value=none]').checked = true; refMode = 'none'; }
  $('contourNote').textContent = photoish ? 'Raise Detail for finer lines. Each feature keeps its own share of detail.' : 'The sample is hand-drawn, so these apply once you choose a photo.';
  for (let p = 1; p < PARTS.length; p++) {
    partPresent[p] = !!(src && src.parts && src.parts.indexOf(p) >= 0);
    const inp = document.querySelector(`#featureChips input[data-part="${p}"]`);
    inp.disabled = !partPresent[p]; inp.checked = partOn[p] && partPresent[p];
    inp.closest('label').classList.toggle('absent', !partPresent[p]);
    inp.closest('label').title = partPresent[p] ? '' : 'Not in the drawing right now';
  }
  const canFocus = !!(src && src.session);
  document.querySelectorAll('[data-tool="focus"]').forEach((b) => { b.disabled = !canFocus; b.title = canFocus ? 'Focus (F): find new contours under the square' : 'Focus needs a photo'; });
  document.querySelectorAll('[data-tool="erase"]').forEach((b) => { b.disabled = !src; });
  document.querySelectorAll('.undoBtn').forEach((b) => { b.disabled = !edits.length; });
  $('btnClearEdits').disabled = !edits.length;
  $('editCount').textContent = edits.length ? `${edits.length} edit${edits.length > 1 ? 's' : ''} applied` : 'No edits yet';
  if (tool && !canFocus && tool === 'focus') setTool('erase');
}
function buildChips() {
  const host = $('featureChips');
  for (const p of CHIP_ORDER) {
    const l = document.createElement('label'); l.className = 'chip part'; l.style.setProperty('--pc', `var(--p${p})`);
    l.innerHTML = `<input type="checkbox" data-part="${p}" checked><span>${PARTS[p].name}</span>`;
    host.appendChild(l);
    l.querySelector('input').addEventListener('change', (e) => { partOn[p] = e.target.checked; onPartsChanged(); });
  }
}

/* ---------- talking to the Python backend ---------- */
async function api(path, body) {
  let r;
  try {
    r = await fetch(path, body instanceof FormData ? { method: 'POST', body }
      : body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  } catch (e) { throw new Error('Could not reach the Python server. Start it with: python app.py'); }
  let j = null;
  try { j = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok || !j || j.error) throw new Error((j && j.error) || `Server error ${r.status}`);
  return j;
}
const toStrokes = (list) => list.map((a) => Float32Array.from(a));
function b64u8(s) { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
function loadImage(url) { return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => rej(new Error('The picture from the server could not be shown.')); im.src = url; }); }

/* strokes -> one closed path -> Fourier coefficients, computed by NumPy on the server */
async function buildCurrent() {
  const vs = visible();
  if (vs.strokes.length < 2) return null;
  const plain = src.kind === 'plain';
  const j = await api('/api/fourier', {
    strokes: vs.strokes.map((st) => Array.from(st, (v) => Math.round(v * 100) / 100)),
    parts: plain ? null : vs.parts, plain, w: src.w, h: src.h
  });
  return { re: Float64Array.from(j.re), im: Float64Array.from(j.im), pen: b64u8(j.pen), part: b64u8(j.part), stats: j.stats };
}

async function present(my, built, opts) {
  lastBuilt = built;
  updateReadout(); computeBoxes(); paintStrokes();
  setStep('features');
  if (opts.message) setStatus(opts.message);
  cStr.style.opacity = 0.95;
  cRef.style.opacity = tool ? 0.38 : 0;
  annoTarget = opts.annotate && !tool ? 1 : 0; annoPupils = 0;
  if (!(await sleep(T(opts.dwell), my))) return;
  setStep('fourier');
  setStatus(`Fourier transform: ${fmt(G)} points into ${fmt(KMAX)} circles`);
  annoTarget = 0;
  applyRefMode();
  if (!kTouched) setCircleCount(autoM(built.stats.length));
  updateReadout();
  startMorph(built, T(opts.quick ? 1000 : 1900));
}

/* ---------- sources ---------- */
async function loadSample() {
  userStarted = true;
  const my = ++seq;
  setStep('src'); setStatus('Asking the Python backend for the sample face');
  let sm;
  try { sm = await api('/api/sample'); } catch (e) { setStatus(e.message, true); return; }
  if (my !== seq) return;
  src = { kind: 'sample', w: sm.w, h: sm.h, session: null, base: { strokes: toStrokes(sm.strokes), parts: sm.parts }, strokes: null, parts: null, analysis: null, refCanvas: null };
  partOn.fill(true); partOn[0] = false;
  edits = []; gesture = null; recomputeFinal();
  updateControls(); repaintStatic();
  $('analysisNote').textContent = 'A hand-drawn sample with every feature.';
  setStatus('Sketching the sample face');
  cRef.style.opacity = 0; cStr.style.opacity = 0;
  await nextFrame();
  if (my !== seq) return;
  let built;
  try { built = await buildCurrent(); } catch (e) { setStatus(e.message, true); return; }
  if (my !== seq) return;
  if (!(await sleep(T(600), my))) return;
  await present(my, built, { quick: false, annotate: true, dwell: 1900, message: 'Hair, ears, eyes, nose, lips, beard: each one is its own group of strokes' });
}

function makeCanvas(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; }

function recomputeFinal(withGesture) {      // automatic strokes + every edit (+ the one being drawn)
  if (!src || !src.base) return;
  const list = withGesture && gesture ? edits.concat([gesture]) : edits;
  const f = list.length ? Core.applyEdits(src.base.strokes, src.base.parts, list) : src.base;
  src.strokes = f.strokes; src.parts = f.parts;
}

/* ask the server to trace the same photo again with the current Detail / Smoothing / Clean-up */
async function retrace() {
  const j = await api('/api/retrace', { session: src.session, detail: params.detail, sigma: params.sigma, min_len: params.minLen });
  src.base = { strokes: toStrokes(j.strokes), parts: j.parts };
  recomputeFinal();
  return src.base.strokes.length > 1;
}

async function loadPhoto(file) {
  const my = ++seq;
  setStep('src'); setStatus('Sending the photo to the Python backend: finding the face and its features');
  cStr.style.opacity = 0; annoTarget = 0;
  const fd = new FormData();
  fd.append('file', file); fd.append('detail', params.detail); fd.append('sigma', params.sigma); fd.append('min_len', params.minLen);
  let j, img;
  try { j = await api('/api/analyze', fd); img = await loadImage(j.image); } catch (e) { setStatus(e.message, true); return; }
  if (my !== seq) return;
  const cv = makeCanvas(j.w, j.h); cv.getContext('2d').drawImage(img, 0, 0);
  edits = []; gesture = null;
  partOn.fill(true); partOn[0] = false;

  if (j.kind === 'plain') {                   // ---- no face: the server traced the whole picture ----
    src = { kind: 'plain', w: j.w, h: j.h, session: j.session, base: { strokes: toStrokes(j.strokes), parts: j.parts }, strokes: null, parts: null, refCanvas: cv, analysis: null };
    updateControls(); repaintStatic();
    $('analysisNote').textContent = 'No face found, so the whole picture is traced. Use Erase to clear what you do not want.';
    cRef.style.opacity = 0.55;
    setStatus('No face found. Tracing the whole picture.');
    if (src.base.strokes.length < 2) { setStatus('No clear edges found. Raise Detail, or try a sharper photo.', true); return; }
    if (!(await sleep(T(700), my))) return;
    recomputeFinal(); updateControls();
    let b;
    try { b = await buildCurrent(); } catch (e) { setStatus(e.message, true); return; }
    if (my !== seq) return;
    await present(my, b, { annotate: false, dwell: 1100, message: 'Contours traced into one closed path' });
    return;
  }

  // ---- face found: the server levelled the eyes, cropped, labelled the features and traced them ----
  src = {
    kind: 'photo', w: j.w, h: j.h, session: j.session, refCanvas: cv, strokes: null, parts: null,
    base: { strokes: toStrokes(j.strokes), parts: j.parts },
    analysis: { pupL: { x: j.pupils.l[0], y: j.pupils.l[1] }, pupR: { x: j.pupils.r[0], y: j.pupils.r[1] }, hasBeard: j.has_beard }
  };
  updateControls();
  paintImage(xRef, src.refCanvas); paintStrokes();
  cRef.style.opacity = 0.9;
  annoPupils = 1;
  setStatus(j.faces > 1 ? `Found ${j.faces} faces, using the most prominent. Eyes levelled, head and shoulders cropped (${j.ms} ms).` : `Found the face and both eyes. Levelled and cropped by the backend in ${j.ms} ms.`);
  if (!(await sleep(T(1300), my))) return;

  setStep('features'); setStatus('Hair, eyes, nose, lips, ears and beard, found separately');
  annoPupils = 0;
  recomputeFinal();
  updateControls();
  if (!src.base.strokes.length) { setStatus('No clear edges found. Raise Detail, or try a sharper photo.', true); return; }
  $('analysisNote').textContent = 'Face found' + (j.faces > 1 ? ` (${j.faces} in the photo, using the most prominent)` : '') + `. Beard: ${j.has_beard ? 'yes' : 'not detected'}. Use Focus and Erase below to fix the lines.`;
  cRef.style.opacity = 0.16;
  let b;
  try { b = await buildCurrent(); } catch (e) { setStatus(e.message, true); return; }
  if (my !== seq) return;
  await present(my, b, { annotate: true, dwell: 2100, message: 'Each feature gets its own strokes. Refine them with Focus and Erase' });
}

async function reprocess() {
  if (!src || !src.session) return;
  const my = ++seq;
  setStatus('Re-tracing on the backend');
  let ok, b;
  try {
    ok = await retrace();
    if (my !== seq) return;
    if (!ok) { setStatus('No clear edges at these settings. Raise Detail.', true); return; }
    updateControls();
    b = await buildCurrent();
  } catch (e) { setStatus(e.message, true); return; }
  if (my !== seq) return;
  if (!b) { setStatus('Nothing left to draw. Turn a feature back on.', true); return; }
  await present(my, b, { annotate: false, dwell: 450, quick: true, message: edits.length ? 'Re-traced. Your edits are kept' : 'Re-traced' });
}
let reprocessTimer = 0;
function scheduleReprocess() { clearTimeout(reprocessTimer); reprocessTimer = setTimeout(reprocess, 240); }

async function onPartsChanged(message) {     // a feature was switched, or contours were edited
  if (!src) return;
  const my = ++seq;
  let b;
  try { b = await buildCurrent(); } catch (e) { setStatus(e.message, true); return; }
  if (my !== seq) return;
  if (!b) { setStatus('Nothing left to draw. Turn a feature back on.', true); return; }
  await present(my, b, { annotate: false, dwell: 350, quick: true, message: typeof message === 'string' ? message : 'Redrawing with your changes' });
}

/* ---------- contour tools: Focus finds new contours under a square, Erase removes them ---------- */
const TOOL_HINT = {
  focus: 'Focus: drag over the picture. New contours are found under the square.',
  erase: 'Erase: drag over lines to rub them out (also removes contours behind the square).'
};
function setTool(name) {
  tool = name === tool ? null : (name || null);
  if (tool === 'focus' && !(src && src.session)) tool = null;
  document.body.classList.toggle('tooling', !!tool);
  document.querySelectorAll('[data-tool]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.tool === tool)));
  document.querySelectorAll('input[name=tool]').forEach((r) => { r.checked = r.value === (tool || 'off'); });
  if (!tool) { cursor = null; gesture = null; }
  applyRefMode();
  setStatus(tool ? TOOL_HINT[tool] : (phase === 'done' ? 'Finished. Replay it, or change the settings.' : 'Drawing'));
}
const toImage = (sx, sy) => { const k = screenScale(); return [(sx - view.ox) / k + src.w / 2 - 0.5, (sy - view.oy) / k + src.h / 2 - 0.5]; };
function updateCursor(e) {
  const r = stage.getBoundingClientRect();
  cursor = { x: e.clientX - r.left, y: e.clientY - r.top };
}
function stepGesture() {
  if (!gesture || !cursor || !src) return;
  const [x, y] = toImage(cursor.x, cursor.y);
  if (lastApplied && Math.hypot(x - lastApplied[0], y - lastApplied[1]) < sizePx * 0.4) return;
  lastApplied = [x, y];
  const g = gesture, h = sizePx / 2, rect = [x - h, y - h, x + h, y + h];
  g.rects.push(rect);
  if (g.type === 'add') {                   // the server finds fresh contours inside this square
    g.pending++;
    api('/api/focus', { session: src.session, rect, level: focusLevel, sigma: params.sigma })
      .then((r) => {
        g.items.push({ rect, strokes: toStrokes(r.strokes), parts: r.parts });
        if (gesture === g) { recomputeFinal(true); paintStrokes(); }
      })
      .catch((e) => setStatus(e.message, true))
      .finally(() => { g.pending--; if (g.closed && !g.pending) finalizeGesture(g); });
  } else { recomputeFinal(true); paintStrokes(); }
}
function commitGesture() {
  const g = gesture; gesture = null; lastApplied = null;
  if (!g || !g.rects.length) return;
  g.closed = true;
  if (!g.pending) finalizeGesture(g);        // Focus squares may still be waiting for the server
}
function finalizeGesture(g) {
  if (g.type === 'add' && !g.items.length) { recomputeFinal(); paintStrokes(); return; }
  edits.push(g);
  recomputeFinal(); paintStrokes();
  updateControls();
  clearTimeout(rebuildTimer);
  rebuildTimer = setTimeout(() => onPartsChanged(), 700);   // several strokes in a row are redrawn together
}
function undoEdit() {
  if (!edits.length) return;
  edits.pop(); recomputeFinal(); paintStrokes(); updateControls();
  clearTimeout(rebuildTimer); rebuildTimer = setTimeout(() => onPartsChanged('Undone'), 500);
}
function clearEdits() {
  if (!edits.length) return;
  edits = []; recomputeFinal(); paintStrokes(); updateControls();
  clearTimeout(rebuildTimer); rebuildTimer = setTimeout(() => onPartsChanged('Edits cleared'), 500);
}
function drawCursor() {
  if (!tool || !cursor || !src) return;
  const c = xLive, k = screenScale(), s = sizePx * k, x0 = cursor.x - s / 2, y0 = cursor.y - s / 2;
  c.save();
  c.beginPath(); c.rect(x0, y0, s, s); c.clip();
  if (tool === 'focus' && src.refCanvas) {          // a lens: the photo shows through, with the contours on top
    const dw = src.w * k, dh = src.h * k;
    c.globalAlpha = 0.96; c.drawImage(src.refCanvas, view.ox - dw / 2, view.oy - dh / 2, dw, dh); c.globalAlpha = 1;
    drawStrokeSet(c);
  } else { c.fillStyle = col.danger; c.globalAlpha = 0.13; c.fillRect(x0, y0, s, s); c.globalAlpha = 1; }
  c.restore();
  c.lineWidth = 1.8; c.strokeStyle = tool === 'focus' ? col.accent : col.danger;
  if (tool === 'erase') c.setLineDash([6, 4]);
  c.strokeRect(x0 + 0.5, y0 + 0.5, s, s);
  c.setLineDash([]);
  const L = Math.min(9, s / 4); c.lineWidth = 2.4;
  c.beginPath();
  c.moveTo(x0 - 3, y0 + L); c.lineTo(x0 - 3, y0 - 3); c.lineTo(x0 + L, y0 - 3);
  c.moveTo(x0 + s + 3 - L, y0 + s + 3); c.lineTo(x0 + s + 3, y0 + s + 3); c.lineTo(x0 + s + 3, y0 + s + 3 - L);
  c.stroke();
}

async function handleFile(file) {
  if (!file) return;
  userStarted = true;
  if (!/^image\//.test(file.type) && !/\.(jpe?g|png|webp|bmp|gif|tiff?)$/i.test(file.name || '')) { setStatus('That file is not an image.', true); return; }
  await loadPhoto(file);
}

/* ---------- saving ---------- */
async function saveImage() {
  if (!src) return;
  if (phase !== 'done') finishNow();
  const k = screenScale(), iw = src.w * k, ih = src.h * k, pad = Math.min(iw, ih) * 0.06;
  const rx = Math.max(0, view.ox - iw / 2 - pad), ry = Math.max(0, view.oy - ih / 2 - pad);
  const rw = Math.min(W - rx, iw + pad * 2), rh = Math.min(H - ry, ih + pad * 2);
  const out = document.createElement('canvas'); out.width = Math.round(rw * DPR); out.height = Math.round(rh * DPR);
  const o = out.getContext('2d');
  o.fillStyle = col.paper; o.fillRect(0, 0, out.width, out.height);
  o.drawImage(cTrail, rx * DPR, ry * DPR, rw * DPR, rh * DPR, 0, 0, out.width, out.height);
  const blob = await new Promise((r) => out.toBlob(r, 'image/png'));
  if (!blob) { setStatus('Could not create the image.', true); return; }
  try {
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'epicycle-portrait.png';
    document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  } catch (e) { if (!e || e.code !== 'declined') setStatus('The image could not be saved here.', true); }
}

/* ---------- animation loop ---------- */
function tick(now) {
  requestAnimationFrame(tick);
  if (!W) return;
  const dt = Math.min(0.05, (now - lastT) / 1000); lastT = now;

  const target = showCircles || showArms ? 1 : 0;
  chainA += (target - chainA) * Math.min(1, dt * 7);
  annoA += (annoTarget - annoA) * Math.min(1, dt * 6);

  if (morph) stepMorph(now);
  if (needRebuild && phase !== 'morph') { needRebuild = false; rebuildTrail(); }

  if (playing) jf += dt * G / drawSecs * (phase === 'morph' ? 0.55 : 1);

  if (phase === 'draw') {
    appendTrail(Math.min(G, Math.floor(jf)));
    if (jf >= G) { phase = 'done'; doneAt = now; jf = 0; if (!tool) setStatus('Finished. Replay it, or change the settings.'); }
    else {
      const pct = Math.floor(jf / G * 100);
      if (pct !== lastPct) { lastPct = pct; if (!tool) setStatus(`Drawing ${pct}%`); }
    }
  } else if (phase === 'done') {
    if (jf >= G) jf -= G;
    if (loopOn && now - doneAt > 2800) { phase = 'fade'; fadeAt = now; cTrail.style.opacity = 0; }
  } else if (phase === 'fade') {
    if (jf >= G) jf -= G;
    if (now - fadeAt > 850) startDrawing();
  } else if (jf >= G) jf -= G;

  elBar.style.transform = `scaleX(${phase === 'draw' ? clamp(jf / G, 0, 1) : phase === 'done' ? 1 : 0})`;

  const c = xLive;
  c.setTransform(DPR, 0, 0, DPR, 0, 0); c.clearRect(0, 0, W, H);
  drawAnnotations();
  if (morph) drawGhost();
  const tip = drawChain((jf % G) / G, chainA * (1 - 0.7 * annoA) * (tool ? 0.5 : 1));
  if (phase === 'draw') drawTail();
  const penA = phase === 'done' || phase === 'fade' ? chainA : 1;
  if (penA > 0.02) drawPen(tip[0], tip[1], penA);
  drawCursor();
}

/* ---------- controls ---------- */
function paintSlider(el) {
  const p = (el.value - el.min) / (el.max - el.min) * 100;
  el.style.setProperty('--p', p + '%');
}
function bind(id, out, onInput, fmtFn) {
  const el = $(id), o = $(out);
  const upd = () => { paintSlider(el); o.textContent = fmtFn(+el.value); };
  el.addEventListener('input', () => { upd(); onInput(+el.value); });
  upd();
}
bind('rDetail', 'oDetail', (v) => { params.detail = v / 100; scheduleReprocess(); }, (v) => v + '%');
bind('rSmooth', 'oSmooth', (v) => { params.sigma = v / 10; scheduleReprocess(); }, (v) => (v / 10).toFixed(1));
bind('rClean', 'oClean', (v) => { params.minLen = v; scheduleReprocess(); }, (v) => v + ' px');
bind('rK', 'oK', (v) => {
  kTouched = true;
  curM = mFromSlider(v); curK = 2 * curM; applyTaper();
  if (phase !== 'morph') { buildOrder(); needRebuild = true; }
  updateReadout();
}, (v) => fmt(2 * mFromSlider(v)));
bind('rTime', 'oTime', (v) => { drawSecs = v; }, (v) => v + ' s');
curM = mFromSlider(+$('rK').value); curK = 2 * curM; applyTaper(); buildOrder();

$('cbCircles').addEventListener('change', (e) => { showCircles = e.target.checked; });
$('cbArms').addEventListener('change', (e) => { showArms = e.target.checked; });
$('cbLoop').addEventListener('change', (e) => { loopOn = e.target.checked; if (loopOn && phase === 'done') doneAt = performance.now() - 1500; });
$('cbColor').addEventListener('change', (e) => { colorParts = e.target.checked; paintStrokes(); paintTrailAll(); });
document.querySelectorAll('[data-tool]').forEach((b) => b.addEventListener('click', () => setTool(b.dataset.tool)));
document.querySelectorAll('input[name=tool]').forEach((r) => r.addEventListener('change', () => { if (r.checked) { r.value === 'off' ? setTool(null) : (tool === r.value ? 0 : setTool(r.value)); } }));
document.querySelectorAll('.undoBtn').forEach((b) => b.addEventListener('click', undoEdit));
$('btnClearEdits').addEventListener('click', clearEdits);
$('cbWhole').addEventListener('change', (e) => { eraseWhole = e.target.checked; });
bind('rSize', 'oSize', (v) => { sizePx = v; }, (v) => v + ' px');
bind('rFocus', 'oFocus', (v) => { focusLevel = v / 100; }, (v) => v + '%');
sizePx = +$('rSize').value; focusLevel = +$('rFocus').value / 100;
stage.addEventListener('pointerdown', (e) => {
  if (!tool || !src || e.button > 0 || e.target.closest('.tools, .steps, .hud')) return;
  e.preventDefault(); stage.setPointerCapture(e.pointerId);
  updateCursor(e);
  gesture = { type: tool === 'focus' ? 'add' : 'erase', rects: [], items: [], pending: 0, closed: false, whole: eraseWhole };
  lastApplied = null; stepGesture();
});
stage.addEventListener('pointermove', (e) => { if (!tool || !src) return; updateCursor(e); if (gesture) stepGesture(); });
const endGesture = (e) => { if (gesture) commitGesture(); if (e.pointerType && e.pointerType !== 'mouse') cursor = null; };
stage.addEventListener('pointerup', endGesture);
stage.addEventListener('pointercancel', endGesture);
stage.addEventListener('pointerleave', () => { if (!gesture) cursor = null; });
stage.addEventListener('wheel', (e) => {
  if (!tool) return;
  e.preventDefault();
  const el = $('rSize'); el.value = clamp(+el.value + (e.deltaY < 0 ? 6 : -6), +el.min, +el.max);
  el.dispatchEvent(new Event('input', { bubbles: true }));
}, { passive: false });
document.querySelectorAll('input[name=ref]').forEach((r) => r.addEventListener('change', () => { refMode = r.value; if (r.checked) applyRefMode(); }));

const btnPlay = $('btnPlay');
function setPlaying(p) { playing = p; btnPlay.textContent = p ? 'Pause' : 'Play'; }
btnPlay.addEventListener('click', () => setPlaying(!playing));
$('btnRedraw').addEventListener('click', () => { if (morph) return; setPlaying(true); startDrawing(); });
$('btnFinish').addEventListener('click', finishNow);
$('btnSave').addEventListener('click', saveImage);
$('btnChoose').addEventListener('click', () => $('file').click());
$('btnSample').addEventListener('click', loadSample);
$('file').addEventListener('change', (e) => { const f = e.target.files && e.target.files[0]; e.target.value = ''; handleFile(f); });

let dragDepth = 0;
window.addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; document.body.classList.add('dragging'); });
window.addEventListener('dragover', (e) => { e.preventDefault(); });
window.addEventListener('dragleave', () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) document.body.classList.remove('dragging'); });
window.addEventListener('drop', (e) => {
  e.preventDefault(); dragDepth = 0; document.body.classList.remove('dragging');
  const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) handleFile(f);
});
window.addEventListener('paste', (e) => {
  const items = (e.clipboardData && e.clipboardData.items) || [];
  for (const it of items) if (it.type && it.type.indexOf('image/') === 0) { handleFile(it.getAsFile()); e.preventDefault(); return; }
});
window.addEventListener('keydown', (e) => {
  const t = e.target;
  const typing = t && (t.tagName === 'INPUT' && t.type !== 'checkbox' && t.type !== 'radio' && t.type !== 'range' || t.tagName === 'TEXTAREA');
  if (typing) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undoEdit(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === 'f') { setTool('focus'); return; }
  if (k === 'e') { setTool('erase'); return; }
  if (k === 'escape' && tool) { setTool(null); return; }
  if (k === '[' || k === ']') { const el = $('rSize'); el.value = clamp(+el.value + (k === ']' ? 6 : -6), +el.min, +el.max); el.dispatchEvent(new Event('input', { bubbles: true })); return; }
  if (e.code !== 'Space' || e.repeat) return;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'BUTTON')) return;
  e.preventDefault(); setPlaying(!playing);
});

/* ---------- boot ---------- */
buildChips();
readColors();
new ResizeObserver(resize).observe(stage);
const onTheme = () => { readColors(); repaintStatic(); };
new MutationObserver(onTheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class', 'style'] });
if (window.matchMedia) {
  const mq = matchMedia('(prefers-color-scheme: dark)');
  if (mq.addEventListener) mq.addEventListener('change', onTheme); else if (mq.addListener) mq.addListener(onTheme);
}
setStep('src');
setStatus('Warming up with one circle');
requestAnimationFrame(tick);
resize();
setTimeout(() => { if (!src && !userStarted) loadSample(); }, T(900));

$('btnSave').hidden = false;

/* handy for testing from the console */
window.__epi = { get phase() { return phase; }, get jf() { return jf; }, loadSample, handleFile, finishNow, get trailN() { return trail.n; }, get src() { return src; }, toScreen, get edits() { return edits; }, get tool() { return tool; } };
})();
