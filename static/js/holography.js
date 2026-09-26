/* ==========================================================================
   holography.js: the Holography page's one tool -- an 8-sample, 1-D
   Gerchberg-Saxton hologram-retrieval loop. Every FFT runs in Python
   (epicycle/holography.py, via numpy.fft) behind /api/holo/target,
   /api/holo/step and /api/holo/reset on this same Flask server; this file
   only asks for numbers and animates them. Scoped in its own IIFE, like
   frequency.js, so it can sit on a page that also loads other tools later.
   ========================================================================== */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const holoWrap = document.querySelector('.holoWrap');
  if (!holoWrap || !$('holoPlay')) return;   // this page/tool isn't present

  const N = 8;
  const RAW_SCALE_MAX = 3.0;
  const reducedMotion = !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);

  /* ---------- talking to the Python backend (same pattern as app.js's api()) ---------- */
  async function api(path, body) {
    let r;
    try {
      r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    } catch (e) {
      setBackendStatus(false);
      throw new Error('Could not reach the Python server. Start it with: python app.py');
    }
    let j = null;
    try { j = await r.json(); } catch (e) { /* not JSON */ }
    if (!r.ok || !j || j.error) { setBackendStatus(false); throw new Error((j && j.error) || `Server error ${r.status}`); }
    setBackendStatus(true);
    return j;
  }
  const apiTarget = (body) => api('/api/holo/target', body);
  const apiStep = (sessionId) => api('/api/holo/step', { session_id: sessionId });
  const apiReset = (sessionId) => api('/api/holo/reset', { session_id: sessionId });

  function setBackendStatus(ok) {
    const el = $('holoBadge'), txt = $('holoBadgeText');
    el.classList.toggle('offline', !ok);
    txt.textContent = ok ? 'Python backend connected (numpy fft)' : "Can't reach the Python server. Start it with: python app.py";
  }

  /* ---------- small local helpers (rendering only, no FFT) ---------- */
  function lerp(a, b, t) { return a + (b - a) * t; }
  function easeInOutCubic(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
  function polar(r, th) { return { re: r * Math.cos(th), im: r * Math.sin(th) }; }
  function hueForPhase(rad) { return ((rad * 180 / Math.PI) + 360) % 360; }
  function colorForPhase(rad, sat, light) { return `hsl(${hueForPhase(rad).toFixed(1)} ${sat}% ${light}%)`; }
  function cssVar(name) { return getComputedStyle(holoWrap).getPropertyValue(name).trim(); }

  /* ---------- state ---------- */
  let sessionId = null, ready = false;
  let targetAmp = new Array(N).fill(0.3);
  let topMax = 1;
  let hologramPhase = new Array(N).fill(0);
  let reconstructedAmpDisplayed = new Array(N).fill(0);
  let errorHistory = [], phaseHistory = [];
  const MAX_HISTORY = 300;
  let iteration = 0, playing = false, animStart = null, iterMs = 1400;
  let prevPhase = hologramPhase.slice();
  let reconTransitionFrom = reconstructedAmpDisplayed.slice();
  let pendingRawAmp = new Array(N).fill(1);
  let pendingRawPhase = new Array(N).fill(0);
  let pendingReconstructedAmp = new Array(N).fill(0);
  let pendingRmse = 0;

  function applyPreview(data) {
    sessionId = data.session_id;
    targetAmp = data.targetAmp;
    topMax = Math.max(...targetAmp) * 1.25;
    hologramPhase = data.hologramPhase;
    reconstructedAmpDisplayed = data.reconstructedAmp;
    pendingRawAmp = data.rawAmp;
    pendingRawPhase = data.rawPhase;
    pendingReconstructedAmp = data.reconstructedAmp;
    pendingRmse = data.rmse;
    iteration = data.iteration;
  }

  async function triggerIteration() {
    if (animStart !== null || !ready) return;
    prevPhase = hologramPhase.slice();
    reconTransitionFrom = reconstructedAmpDisplayed.slice();
    setControlsDisabled(true);
    try {
      const data = await apiStep(sessionId);
      pendingRawAmp = data.rawAmp;
      pendingRawPhase = data.rawPhase;
      pendingReconstructedAmp = data.reconstructedAmp;
      pendingRmse = data.rmse;
      animStart = performance.now();
    } catch (e) {
      setUploadNote(e.message, true);
      setControlsDisabled(false);
      playing = false; holoPlay.textContent = 'Play';
    }
  }

  function commitIteration() {
    hologramPhase = pendingRawPhase.slice();
    reconstructedAmpDisplayed = pendingReconstructedAmp.slice();
    iteration++;
    errorHistory.push(pendingRmse);
    if (errorHistory.length > 200) errorHistory.shift();
    phaseHistory.push(hologramPhase.slice());
    if (phaseHistory.length > MAX_HISTORY) phaseHistory.shift();
    renderHistory();
    animStart = null;
    updateStatsUI();
    setControlsDisabled(false);
    if (playing) triggerIteration();
  }

  async function resetHologram(keepHistory) {
    if (animStart !== null || !ready) return;
    if (!keepHistory) { phaseHistory = []; renderHistory(); }
    setControlsDisabled(true);
    try {
      const data = await apiReset(sessionId);
      applyPreview(data);
      errorHistory = [];
      updateStatsUI();
    } catch (e) { setUploadNote(e.message, true); }
    setControlsDisabled(false);
  }

  async function setPattern(name) {
    if (animStart !== null) return;
    holoTargetThumb.style.display = 'none';
    holoCtrlThumb.style.display = 'none';
    setUploadNote('');
    setControlsDisabled(true);
    try {
      const data = await apiTarget({ pattern: name });
      applyPreview(data);
      errorHistory = []; phaseHistory = [];
      renderHistory();
      ready = true;
      updateStatsUI();
    } catch (e) { setUploadNote(e.message, true); }
    setControlsDisabled(false);
  }

  /* ---------- image -> target amplitude ----------
     Downsampling an image to 8 raw brightness values is plain pixel sampling,
     not FFT work, so it stays in the browser; the backend still does the
     energy normalization and every GS iteration past that point. Every step
     that can fail (a broken/undecodable file, a tainted canvas, a network
     hiccup on the follow-up request) is caught here and reported in
     #holoUploadNote instead of failing silently. */
  function imageToRawShape(img) {
    const off = document.createElement('canvas');
    off.width = N; off.height = 1;
    const octx = off.getContext('2d');
    octx.drawImage(img, 0, 0, N, 1);
    const data = octx.getImageData(0, 0, N, 1).data;   // throws if the canvas got tainted
    const shape = [];
    for (let k = 0; k < N; k++) {
      const r = data[k * 4], g = data[k * 4 + 1], b = data[k * 4 + 2];
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b; // 0..255
      shape.push(0.15 + (lum / 255) * 4.5);
    }
    return shape;
  }

  async function setCustomShape(shape) {
    if (animStart !== null) return;
    setControlsDisabled(true);
    try {
      const data = await apiTarget({ customShape: shape });
      applyPreview(data);
      errorHistory = []; phaseHistory = [];
      renderHistory();
      ready = true;
      updateStatsUI();
    } catch (e) { setUploadNote(e.message, true); }
    setControlsDisabled(false);
  }

  function setUploadNote(msg, isErr) {
    const el = $('holoUploadNote');
    el.textContent = msg || '';
    el.classList.toggle('err', !!isErr);
  }

  /* ---------- DOM refs ---------- */
  const holoPlay = $('holoPlay'), holoStep = $('holoStep'), holoReset = $('holoReset');
  const speed = $('holoSpeed'), iterVal = $('holoIter'), errVal = $('holoErr'), convTag = $('holoConv');
  const patternBtns = Array.from(document.querySelectorAll('.holoControls [data-pattern]'));
  const imgInput = $('holoImgInput'), holoCtrlThumb = $('holoCtrlThumb'), holoTargetThumb = $('holoTargetThumb');

  const ampTargetCanvas = $('holoAmpTarget'), rawAmpCanvas = $('holoRawAmp'), sparkCanvas = $('holoSpark');
  const phasorRow = $('holoPhasorRow'), waveRow = $('holoWaveRow'), phaseStripRow = $('holoPhaseRow');
  const historyScroll = $('holoHistScroll'), historyEmpty = $('holoHistEmpty'), historyCanvas = $('holoHistCanvas');

  const phasorCanvases = [], phasorLabels = [];
  for (let k = 0; k < N; k++) {
    const cell = document.createElement('div'); cell.className = 'holoCell';
    const cv = document.createElement('canvas'); cell.appendChild(cv);
    const kl = document.createElement('div'); kl.className = 'holoK'; kl.textContent = 'k = ' + k; cell.appendChild(kl);
    const rv = document.createElement('div'); rv.className = 'holoRv';
    rv.innerHTML = '<span class="a">A —</span><br><span class="p">&phi; —</span>';
    cell.appendChild(rv);
    phasorRow.appendChild(cell);
    phasorCanvases.push(cv); phasorLabels.push(rv);
  }
  const waveCanvases = [];
  for (let k = 0; k < N; k++) {
    const cell = document.createElement('div'); cell.className = 'holoCell holoWave';
    const cv = document.createElement('canvas'); cell.appendChild(cv);
    const kl = document.createElement('div'); kl.className = 'holoK'; kl.textContent = 'k = ' + k; cell.appendChild(kl);
    waveRow.appendChild(cell);
    waveCanvases.push(cv);
  }
  const phaseCells = [];
  for (let k = 0; k < N; k++) {
    const cell = document.createElement('div'); cell.className = 'holoPhaseCell';
    const sw = document.createElement('div'); sw.className = 'holoSwCell'; cell.appendChild(sw);
    const deg = document.createElement('div'); deg.className = 'holoDeg'; deg.textContent = '—'; cell.appendChild(deg);
    phaseStripRow.appendChild(cell);
    phaseCells.push({ sw, deg });
  }

  function setControlsDisabled(disabled) {
    holoStep.disabled = disabled;
    holoReset.disabled = disabled;
    patternBtns.forEach((b) => { b.disabled = disabled; });
    imgInput.disabled = disabled;
  }
  function updateStatsUI() {
    iterVal.textContent = iteration;
    const shown = errorHistory.length ? errorHistory[errorHistory.length - 1] : pendingRmse;
    errVal.textContent = shown.toFixed(3);
    convTag.classList.toggle('show', shown < 0.12);
  }
  updateStatsUI();

  imgInput.addEventListener('change', (e) => {
    const file = e.target.files && e.target.files[0];
    e.target.value = '';                       // lets the same file be re-picked after an error
    if (!file) return;
    if (!/^image\//.test(file.type) && !/\.(jpe?g|png|webp|bmp|gif|avif)$/i.test(file.name || '')) {
      setUploadNote('That file is not an image.', true);
      return;
    }
    setUploadNote('Reading image…');
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onerror = () => { setUploadNote('That image could not be decoded.', true); URL.revokeObjectURL(url); };
    img.onload = () => {
      if (playing) { playing = false; holoPlay.textContent = 'Play'; }
      let shape;
      try {
        shape = imageToRawShape(img);
      } catch (err) {
        setUploadNote('Could not read this image (try a different file or format).', true);
        URL.revokeObjectURL(url);
        return;
      }
      patternBtns.forEach((b) => b.setAttribute('aria-pressed', 'false'));
      holoCtrlThumb.src = url; holoCtrlThumb.style.display = 'inline-block';
      holoTargetThumb.src = url; holoTargetThumb.style.display = 'block';
      setCustomShape(shape).then(() => {
        setUploadNote(sessionId ? '' : 'The image target could not be reached.');
        URL.revokeObjectURL(url);
      });
    };
    img.src = url;
  });

  /* ---------- canvas drawing ---------- */
  function fitCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: rect.width, h: rect.height };
  }

  function drawGroupedBars(canvas, target, recon, max) {
    const { ctx, w, h } = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    const padB = 18, padT = 6, padX = 6;
    const chartH = h - padB - padT;
    const groupW = (w - padX * 2) / N;
    const barW = groupW * 0.34;
    const textDim = cssVar('--muted'), targetCol = cssVar('--holo-target'), ampCol = cssVar('--holo-amp');

    ctx.strokeStyle = cssVar('--line'); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padX, padT + chartH + 0.5); ctx.lineTo(w - padX, padT + chartH + 0.5); ctx.stroke();

    for (let k = 0; k < N; k++) {
      const gx = padX + k * groupW;
      const th = Math.max(1, (target[k] / max) * chartH);
      const rh = Math.max(1, (Math.min(recon[k], max * 1.3) / max) * chartH);
      ctx.fillStyle = targetCol; ctx.globalAlpha = 0.55;
      ctx.fillRect(gx + groupW * 0.12, padT + chartH - th, barW, th);
      ctx.globalAlpha = 1;
      ctx.fillStyle = ampCol;
      ctx.fillRect(gx + groupW * 0.12 + barW + 3, padT + chartH - rh, barW, rh);
      ctx.fillStyle = textDim; ctx.font = '10px ' + cssVar('--sans'); ctx.textAlign = 'center';
      ctx.fillText(k, gx + groupW / 2, h - 4);
    }
  }

  function drawRefBars(canvas, values, max, refLine) {
    const { ctx, w, h } = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    const padB = 18, padT = 6, padX = 6;
    const chartH = h - padB - padT;
    const groupW = (w - padX * 2) / N;
    const barW = groupW * 0.5;
    const textDim = cssVar('--muted'), ampCol = cssVar('--holo-amp'), targetCol = cssVar('--holo-target');

    const refY = padT + chartH - (refLine / max) * chartH;
    ctx.strokeStyle = targetCol; ctx.globalAlpha = 0.7; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(padX, refY); ctx.lineTo(w - padX, refY); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;

    ctx.strokeStyle = cssVar('--line');
    ctx.beginPath(); ctx.moveTo(padX, padT + chartH + 0.5); ctx.lineTo(w - padX, padT + chartH + 0.5); ctx.stroke();

    for (let k = 0; k < N; k++) {
      const gx = padX + k * groupW;
      const v = Math.min(values[k], max * 1.3);
      const bh = Math.max(1, (v / max) * chartH);
      ctx.fillStyle = ampCol; ctx.globalAlpha = 0.85;
      ctx.fillRect(gx + (groupW - barW) / 2, padT + chartH - bh, barW, bh);
      ctx.globalAlpha = 1;
      ctx.fillStyle = textDim; ctx.font = '10px ' + cssVar('--sans'); ctx.textAlign = 'center';
      ctx.fillText(k, gx + groupW / 2, h - 4);
    }
  }

  function drawPhasor(canvas, amp, phase) {
    const { ctx, w, h } = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2;
    const maxR = Math.min(w, h) / 2 - 6;
    const unitR = maxR / RAW_SCALE_MAX;
    const border = cssVar('--line'), textDim = cssVar('--muted');

    ctx.strokeStyle = border; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.arc(cx, cy, unitR, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(cx, cy, maxR, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = border; ctx.globalAlpha = 0.6;
    ctx.beginPath(); ctx.moveTo(cx - maxR, cy); ctx.lineTo(cx + maxR, cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, cy - maxR); ctx.lineTo(cx, cy + maxR); ctx.stroke();
    ctx.globalAlpha = 1;

    const r = Math.min(amp, RAW_SCALE_MAX) * unitR;
    const x = cx + r * Math.cos(phase), y = cy - r * Math.sin(phase);
    const col = colorForPhase(phase, 72, 42);

    ctx.strokeStyle = col; ctx.lineWidth = 2.4;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
    ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();

    ctx.fillStyle = textDim; ctx.font = '9px ' + cssVar('--sans'); ctx.textAlign = 'center';
    ctx.fillText('1', cx + unitR + 8, cy + 3);
  }

  function drawTimeWave(canvas, amp, phase, t) {
    const { ctx, w, h } = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    const midY = h / 2;
    const ampScale = (h / 2 - 5) / RAW_SCALE_MAX;
    const periodPx = w / 2.2;

    ctx.strokeStyle = cssVar('--line'); ctx.globalAlpha = 0.7;
    ctx.beginPath(); ctx.moveTo(0, midY); ctx.lineTo(w, midY); ctx.stroke();
    ctx.globalAlpha = 1;

    const col = colorForPhase(phase, 72, 42);
    ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let x = 0; x <= w; x += 2) {
      const theta = (x / periodPx) * 2 * Math.PI - t + phase;
      const y = midY - Math.min(amp, RAW_SCALE_MAX) * ampScale * Math.sin(theta);
      if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function drawSparkline(canvas, history, live) {
    const { ctx, w, h } = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    const padX = 6, padY = 8;
    const data = history.slice();
    if (live !== undefined) data.push(live);
    if (data.length < 2) {
      ctx.fillStyle = cssVar('--muted'); ctx.font = '11px ' + cssVar('--sans'); ctx.textAlign = 'center';
      ctx.fillText('waiting for iterations…', w / 2, h / 2);
      return;
    }
    const max = Math.max(...data, 0.05), min = 0;
    const stepX = (w - padX * 2) / (data.length - 1);
    ctx.strokeStyle = cssVar('--holo-amp'); ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = padX + i * stepX, y = padY + (1 - (v - min) / (max - min)) * (h - padY * 2);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.lineTo(padX + (data.length - 1) * stepX, h - padY);
    ctx.lineTo(padX, h - padY);
    ctx.closePath();
    ctx.fillStyle = cssVar('--holo-amp-soft'); ctx.fill();
  }

  function renderPhaseStrip(phases) {
    for (let k = 0; k < N; k++) {
      phaseCells[k].sw.style.background = colorForPhase(phases[k], 60, 46);
      phaseCells[k].deg.textContent = Math.round(phases[k] * 180 / Math.PI) + '\u00B0';
    }
  }
  function renderPhasorLabels(amps, phases) {
    for (let k = 0; k < N; k++) {
      const deg = Math.round(phases[k] * 180 / Math.PI);
      phasorLabels[k].innerHTML = '<span class="a">A ' + amps[k].toFixed(2) + '</span><br><span class="p">&phi; ' + deg + '\u00B0</span>';
    }
  }

  function renderHistory() {
    if (phaseHistory.length === 0) {
      historyEmpty.style.display = 'flex';
      historyCanvas.style.display = 'none';
      return;
    }
    historyEmpty.style.display = 'none';
    historyCanvas.style.display = 'block';
    const rowH = 16, labelW = 56;
    const cssW = historyScroll.clientWidth || 400;
    const cssH = phaseHistory.length * rowH;
    const dpr = window.devicePixelRatio || 1;
    historyCanvas.style.width = cssW + 'px';
    historyCanvas.style.height = cssH + 'px';
    historyCanvas.width = Math.round(cssW * dpr);
    historyCanvas.height = Math.round(cssH * dpr);
    const ctx = historyCanvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    const cellW = (cssW - labelW - 10) / N;
    const textDim = cssVar('--muted');
    ctx.font = '10px ' + cssVar('--sans');
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';

    phaseHistory.forEach((row, i) => {
      const y = i * rowH;
      ctx.fillStyle = textDim;
      ctx.fillText('iter ' + (i + 1), 4, y + rowH / 2);
      for (let k = 0; k < N; k++) {
        ctx.fillStyle = colorForPhase(row[k], 55, 50);
        ctx.fillRect(labelW + k * cellW, y + 1, cellW - 1.5, rowH - 2);
      }
    });
    historyScroll.scrollTop = historyScroll.scrollHeight;
  }

  /* ---------- main animation loop ---------- */
  let waveClock = 0, lastTs = null;
  function tick(now) {
    if (lastTs === null) lastTs = now;
    const dt = (now - lastTs) / 1000;
    lastTs = now;
    if (!reducedMotion) waveClock += dt * 1.6;

    let curAmp, curPhase;
    if (animStart !== null) {
      let t = (now - animStart) / iterMs;
      if (reducedMotion) t = 1;
      if (t >= 1) t = 1;
      const e = easeInOutCubic(t);

      curAmp = new Array(N); curPhase = new Array(N);
      for (let k = 0; k < N; k++) {
        let re, im;
        if (e < 0.5) {
          const lt = e / 0.5;
          const startC = polar(1, prevPhase[k]);
          const midC = polar(pendingRawAmp[k], pendingRawPhase[k]);
          re = lerp(startC.re, midC.re, lt); im = lerp(startC.im, midC.im, lt);
        } else {
          const lt = (e - 0.5) / 0.5;
          const midC = polar(pendingRawAmp[k], pendingRawPhase[k]);
          const endC = polar(1, pendingRawPhase[k]);
          re = lerp(midC.re, endC.re, lt); im = lerp(midC.im, endC.im, lt);
        }
        curAmp[k] = Math.hypot(re, im); curPhase[k] = Math.atan2(im, re);
      }
      const curRecon = reconTransitionFrom.map((v, k) => lerp(v, pendingReconstructedAmp[k], e));

      drawGroupedBars(ampTargetCanvas, targetAmp, curRecon, topMax);
      drawRefBars(rawAmpCanvas, curAmp, RAW_SCALE_MAX, 1);
      renderPhasorLabels(curAmp, curPhase);
      renderPhaseStrip(curPhase);
      drawSparkline(sparkCanvas, errorHistory, pendingRmse);

      if (t >= 1) commitIteration();
    } else {
      curAmp = hologramPhase.map(() => 1);
      curPhase = hologramPhase;
      drawGroupedBars(ampTargetCanvas, targetAmp, reconstructedAmpDisplayed, topMax);
      drawRefBars(rawAmpCanvas, curAmp, RAW_SCALE_MAX, 1);
      renderPhasorLabels(curAmp, curPhase);
      renderPhaseStrip(curPhase);
      drawSparkline(sparkCanvas, errorHistory);
    }

    for (let k = 0; k < N; k++) {
      drawPhasor(phasorCanvases[k], curAmp[k], curPhase[k]);
      drawTimeWave(waveCanvases[k], curAmp[k], curPhase[k], waveClock);
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  /* ---------- controls wiring ---------- */
  holoPlay.addEventListener('click', () => {
    playing = !playing;
    holoPlay.textContent = playing ? 'Pause' : 'Play';
    if (playing) triggerIteration();
  });
  holoStep.addEventListener('click', () => {
    if (playing) { playing = false; holoPlay.textContent = 'Play'; }
    triggerIteration();
  });
  holoReset.addEventListener('click', () => {
    if (playing) { playing = false; holoPlay.textContent = 'Play'; }
    resetHologram(false);
  });
  speed.addEventListener('input', () => { iterMs = 3000 - Number(speed.value) * 260; });
  iterMs = 3000 - Number(speed.value) * 260;

  patternBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      if (playing) { playing = false; holoPlay.textContent = 'Play'; }
      patternBtns.forEach((b) => b.setAttribute('aria-pressed', 'false'));
      btn.setAttribute('aria-pressed', 'true');
      setPattern(btn.dataset.pattern);
    });
  });

  window.addEventListener('resize', renderHistory);
  document.addEventListener('fpi-themechange', () => { renderHistory(); });

  /* ---------- init ---------- */
  setPattern('single');
})();