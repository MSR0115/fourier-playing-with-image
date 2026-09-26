// Fourier hologram generator — the 2nd holography tool.
// Calls /api/gsgen/target, /api/gsgen/step, /api/gsgen/reset.
// Adapted from the standalone GerchbergSaxton project's index.html script,
// scoped to the [data-tool="gsgen"] panel inside holography.html.
(function () {
  "use strict";

  const panel = document.querySelector('.holoToolPanel[data-tool="gsgen"]');
  if (!panel) return;

  let sessionId = null;
  let iteration = 0;
  let history = [];
  let playing = false;
  let playTimer = null;
  let lastMode = "text";
  let lastShape = "circle";
  let lastFile = null;

  // ---- DOM ----
  const imgInput      = panel.querySelector("#gsgen-imgInput");
  const textInput     = panel.querySelector("#gsgen-textInput");
  const textBtn       = panel.querySelector("#gsgen-textBtn");
  const shapeBtns     = Array.from(panel.querySelectorAll("[data-shape]"));
  const resSelect     = panel.querySelector("#gsgen-resSelect");
  const playBtn       = panel.querySelector("#gsgen-play");
  const stepBtn       = panel.querySelector("#gsgen-step");
  const resetBtn      = panel.querySelector("#gsgen-reset");
  const speedInput    = panel.querySelector("#gsgen-speed");
  const iterVal       = panel.querySelector("#gsgen-iterVal");
  const corrVal       = panel.querySelector("#gsgen-corrVal");
  const timeVal       = panel.querySelector("#gsgen-timeVal");
  const convTag       = panel.querySelector("#gsgen-convTag");
  const targetImg     = panel.querySelector("#gsgen-targetImg");
  const targetEmpty   = panel.querySelector("#gsgen-targetEmpty");
  const hologramImg   = panel.querySelector("#gsgen-hologramImg");
  const reconImg      = panel.querySelector("#gsgen-reconImg");
  const sparkCanvas   = panel.querySelector("#gsgen-sparkCanvas");
  const downloadHolo  = panel.querySelector("#gsgen-downloadHolo");
  const downloadRecon = panel.querySelector("#gsgen-downloadRecon");
  const opticsRow     = panel.querySelector("#gsgen-opticsRow");
  const slmMini       = panel.querySelector("#gsgen-slmMini");
  const screenMini    = panel.querySelector("#gsgen-screenMini");

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // ---- optics diagram ----
  const ICONS = {
    laser: '<svg viewBox="0 0 40 40" width="30" height="30" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="14" width="15" height="12" rx="2"/><line x1="19" y1="20" x2="34" y2="20"/><polygon points="28,15 35,20 28,25" fill="currentColor" stroke="none"/></svg>',
    lens:  '<svg viewBox="0 0 40 40" width="30" height="30" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 4 C29 4 29 36 20 36 C11 36 11 4 20 4 Z"/><line x1="20" y1="1" x2="20" y2="39" stroke-dasharray="2 3" stroke-width="1"/></svg>'
  };
  function icon(svg) { const d = document.createElement("div"); d.innerHTML = svg; return d.firstElementChild; }
  function makeStage(title, desc, opts) {
    const stage = document.createElement("div"); stage.className = "gsgen-stage";
    const box = document.createElement("div"); box.className = "gsgen-stageIcon";
    if (opts && opts.miniImgId) { const im = document.createElement("img"); im.id = opts.miniImgId; im.alt = title; box.appendChild(im); }
    else if (opts && opts.iconKey) { box.appendChild(icon(ICONS[opts.iconKey])); }
    stage.appendChild(box);
    const t = document.createElement("div"); t.className = "gsgen-stageTitle"; t.textContent = title; stage.appendChild(t);
    const d = document.createElement("div"); d.className = "gsgen-stageDesc"; d.textContent = desc; stage.appendChild(d);
    return stage;
  }
  function makeConn() { const s = document.createElement("div"); s.className = "gsgen-conn"; s.textContent = "\u2192"; return s; }

  if (opticsRow) {
    opticsRow.appendChild(makeStage("Laser", "Coherent, collimated beam illuminates the setup", { iconKey: "laser" }));
    opticsRow.appendChild(makeConn());
    opticsRow.appendChild(makeStage("Beam expander", "Widens the beam to cover the whole SLM", { iconKey: "lens" }));
    opticsRow.appendChild(makeConn());
    opticsRow.appendChild(makeStage("Spatial light modulator", "Displays the computed phase pattern", { miniImgId: "gsgen-slmMini" }));
    opticsRow.appendChild(makeConn());
    opticsRow.appendChild(makeStage("Fourier lens", "Performs a real optical Fourier transform", { iconKey: "lens" }));
    opticsRow.appendChild(makeConn());
    opticsRow.appendChild(makeStage("Screen / camera", "The reconstructed image appears at the focal plane", { miniImgId: "gsgen-screenMini" }));
  }

  // ---- sparkline ----
  function drawSparkline() {
    if (!sparkCanvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = sparkCanvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (sparkCanvas.width !== w || sparkCanvas.height !== h) {
      sparkCanvas.width = w; sparkCanvas.height = h;
    }
    const ctx = sparkCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    const padX = 6, padY = 8;
    if (history.length < 2) {
      ctx.fillStyle = cssVar("--muted") || "#888";
      ctx.font = '11px "IBM Plex Mono", monospace';
      ctx.textAlign = "center";
      ctx.fillText("waiting for iterations\u2026", rect.width / 2, rect.height / 2);
      return;
    }
    const stepX = (rect.width - padX * 2) / (history.length - 1);
    ctx.strokeStyle = cssVar("--holo-amp") || "#6AA6FF";
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach((v, i) => {
      const x = padX + i * stepX;
      const y = padY + (1 - v) * (rect.height - padY * 2);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.lineTo(padX + (history.length - 1) * stepX, rect.height - padY);
    ctx.lineTo(padX, rect.height - padY);
    ctx.closePath();
    ctx.fillStyle = "rgba(106,166,255,0.12)";
    ctx.fill();
  }

  function updateStats(corr) {
    if (iterVal)  iterVal.textContent = iteration;
    if (corrVal)  corrVal.textContent = (corr * 100).toFixed(1) + "%";
    if (convTag)  convTag.classList.toggle("show", corr > 0.9);
  }

  function setImages(hologramSrc, reconSrc) {
    if (hologramImg) hologramImg.src = hologramSrc;
    if (reconImg)    reconImg.src = reconSrc;
    const sm = panel.querySelector("#gsgen-slmMini");
    const scr = panel.querySelector("#gsgen-screenMini");
    if (sm)  sm.src = hologramSrc;
    if (scr) scr.src = reconSrc;
  }

  // ---- backend calls ----
  async function callTarget(formData) {
    const t0 = performance.now();
    try {
      const res = await fetch("/api/gsgen/target", { method: "POST", body: formData });
      if (!res.ok) return;
      const data = await res.json();
      sessionId = data.session_id;
      iteration = data.iteration;
      history = [];
      if (targetImg) { targetImg.src = data.target_image; targetImg.style.display = ""; }
      if (targetEmpty) targetEmpty.style.display = "none";
      setImages(data.hologram_phase, data.reconstruction);
      updateStats(data.preview_correlation || 0);
      drawSparkline();
      if (timeVal) timeVal.textContent = (performance.now() - t0).toFixed(0);
    } catch (e) { console.error("gsgen target error", e); }
  }

  async function stepOnce() {
    if (!sessionId) return;
    const t0 = performance.now();
    try {
      const res = await fetch("/api/gsgen/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId })
      });
      if (!res.ok) { stopPlaying(); return; }
      const data = await res.json();
      iteration = data.iteration;
      history.push(data.correlation);
      if (history.length > 300) history.shift();
      setImages(data.hologram_phase, data.reconstruction);
      updateStats(data.correlation);
      drawSparkline();
      if (timeVal) timeVal.textContent = (performance.now() - t0).toFixed(0);
    } catch (e) { stopPlaying(); }
  }

  async function resetHologram() {
    stopPlaying();
    if (!sessionId) return;
    try {
      const res = await fetch("/api/gsgen/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId })
      });
      if (!res.ok) return;
      const data = await res.json();
      iteration = data.iteration;
      history = [];
      setImages(data.hologram_phase, data.reconstruction);
      updateStats(data.preview_correlation || 0);
      drawSparkline();
    } catch (e) { console.error("gsgen reset error", e); }
  }

  function stopPlaying() {
    playing = false;
    if (playBtn) playBtn.textContent = "Play";
    if (playTimer) { clearTimeout(playTimer); playTimer = null; }
  }
  function playLoop() {
    if (!playing) return;
    stepOnce().finally(() => {
      if (!playing) return;
      const delay = Math.max(30, 900 - Number(speedInput?.value || 5) * 80);
      playTimer = setTimeout(playLoop, delay);
    });
  }

  // ---- source controls ----
  function currentSizeForm() {
    const fd = new FormData();
    fd.append("size", resSelect ? resSelect.value : "128");
    return fd;
  }
  function regenerateTarget() {
    stopPlaying();
    const fd = currentSizeForm();
    if (lastMode === "text") {
      fd.append("mode", "text");
      fd.append("text", (textInput ? textInput.value.trim() : "") || "HOLOGRAM");
    } else if (lastMode === "shape") {
      fd.append("mode", "shape");
      fd.append("shape", lastShape);
    } else if (lastMode === "image" && lastFile) {
      fd.append("mode", "image");
      fd.append("image", lastFile);
    } else return;
    callTarget(fd);
  }

  if (textBtn) textBtn.addEventListener("click", () => {
    lastMode = "text";
    shapeBtns.forEach(b => b.setAttribute("aria-pressed", "false"));
    regenerateTarget();
  });
  if (textInput) textInput.addEventListener("keydown", e => { if (e.key === "Enter") textBtn?.click(); });

  shapeBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      lastMode = "shape"; lastShape = btn.dataset.shape;
      shapeBtns.forEach(b => b.setAttribute("aria-pressed", String(b === btn)));
      regenerateTarget();
    });
  });

  if (imgInput) imgInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    lastMode = "image"; lastFile = file;
    shapeBtns.forEach(b => b.setAttribute("aria-pressed", "false"));
    regenerateTarget();
  });

  if (resSelect) resSelect.addEventListener("change", regenerateTarget);

  if (playBtn) playBtn.addEventListener("click", () => {
    if (playing) { stopPlaying(); }
    else { playing = true; playBtn.textContent = "Pause"; playLoop(); }
  });
  if (stepBtn) stepBtn.addEventListener("click", () => { stopPlaying(); stepOnce(); });
  if (resetBtn) resetBtn.addEventListener("click", resetHologram);

  function triggerDownload(img, filename) {
    if (!img || !img.src) return;
    const a = document.createElement("a");
    a.href = img.src; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
  }
  if (downloadHolo)  downloadHolo.addEventListener("click", () => triggerDownload(hologramImg, "hologram-phase.png"));
  if (downloadRecon) downloadRecon.addEventListener("click", () => triggerDownload(reconImg, "reconstructed-image.png"));

  window.addEventListener("resize", drawSparkline);

  // ---- auto-load default target ----
  regenerateTarget();
})();
