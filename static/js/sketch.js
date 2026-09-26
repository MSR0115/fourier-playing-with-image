/* ==========================================================================
   sketch.js: "Fourier sketching" panel — grayscale -> Fourier high-pass edge
   detection -> thresholding -> pencil shading. Talks to /api/sketch on the
   same Flask backend as the rest of the hub (epicycle/sketch.py).
   Loaded alongside app.js and frequency.js on photography.html — scoped in
   its own IIFE with its own $, every id here is prefixed sk- so nothing
   collides with the other panels.
   ========================================================================== */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  if (!$('sk-file')) return;   // panel not present on this page

  function makeDebouncedRunner(delay) {
    let timer = null, controller = null;
    return (fn) => {
      clearTimeout(timer);
      timer = setTimeout(async () => {
        if (controller) controller.abort();
        controller = new AbortController();
        try { await fn(controller.signal); }
        catch (err) { if (err.name !== 'AbortError') console.error(err); }
      }, delay);
    };
  }
  const skRun = makeDebouncedRunner(450);

  function setSpinner(ids, on) {
    ids.forEach((id) => { const el = $(id); if (el) el.classList.toggle('active', on); });
  }
  const SPINNERS = ['sk-spin1', 'sk-spin2', 'sk-spin3', 'sk-spin4', 'sk-spin5'];

  /* ---------- dropzone ---------- */
  const drop = $('sk-drop'), fileInput = $('sk-file'), filename = $('sk-filename');
  drop.addEventListener('click', (e) => { if (e.target !== fileInput) fileInput.click(); });
  drop.addEventListener('dragover', (e) => e.preventDefault());
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      fileInput.files = e.dataTransfer.files;
      onFileChange();
    }
  });
  fileInput.addEventListener('change', onFileChange);
  function onFileChange() {
    filename.textContent = fileInput.files[0] ? fileInput.files[0].name : 'Drop image or click to browse';
    skRun(process);
  }

  /* ---------- controls ---------- */
  $('sk-threshmethod').addEventListener('change', () => {
    $('sk-manualwrap').style.display = $('sk-threshmethod').value === 'manual' ? 'block' : 'none';
    skRun(process);
  });
  $('sk-hpmode').addEventListener('change', () => skRun(process));
  $('sk-shading').addEventListener('change', () => skRun(process));
  ['sk-cutoff', 'sk-manualthresh', 'sk-presmooth', 'sk-blursigma'].forEach((id) => {
    $(id).addEventListener('input', () => { $(id + 'val').textContent = $(id).value; skRun(process); });
  });

  /* ---------- talking to the backend ---------- */
  let lastFinalSrc = '';
  async function process(signal) {
    if (!fileInput.files[0]) return;
    $('sk-status').textContent = 'Processing\u2026';
    setSpinner(SPINNERS, true);

    const fd = new FormData();
    fd.append('image', fileInput.files[0]);
    fd.append('cutoff', $('sk-cutoff').value);
    fd.append('hp_mode', $('sk-hpmode').value);
    fd.append('thresh_method', $('sk-threshmethod').value);
    fd.append('manual_thresh', $('sk-manualthresh').value);
    fd.append('pre_smooth', $('sk-presmooth').value);
    fd.append('use_shading', $('sk-shading').checked ? 'true' : 'false');
    fd.append('blur_sigma', $('sk-blursigma').value);

    try {
      const res = await fetch('/api/sketch', { method: 'POST', body: fd, signal });
      const data = await res.json();
      setSpinner(SPINNERS, false);
      if (!res.ok) { $('sk-status').textContent = 'Error: ' + data.error; return; }
      $('sk-imgGray').src = data.gray;
      $('sk-imgSmooth').src = data.smoothed;
      $('sk-imgEdge').src = data.edge;
      $('sk-imgThresh').src = data.threshold;
      $('sk-imgFinal').src = lastFinalSrc = data.final;
      $('sk-status').textContent = 'Done.';
    } catch (err) {
      if (err.name !== 'AbortError') {
        setSpinner(SPINNERS, false);
        $('sk-status').textContent = 'Request failed: ' + err.message;
      }
      throw err;
    }
  }

  $('sk-download').addEventListener('click', () => {
    if (!lastFinalSrc) return;
    const a = document.createElement('a');
    a.href = lastFinalSrc;
    a.download = 'fourier-sketch.png';
    document.body.appendChild(a); a.click(); a.remove();
  });
})();