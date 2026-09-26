/* ==========================================================================
   frequency.js: "Frequency filtering" (single-image CFT filters) and
   "Hybrid Image" (two-source hybrid) panels. Talks to /api/process and
   /api/hybrid on the same Flask backend the epicycle tool uses.
   Loaded alongside app.js on photography.html — scoped in its own IIFE with
   its own $, and every id here is prefixed fq- / hy- so nothing collides.
   ========================================================================== */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  if (!$('fq-file')) return;   // panels not present on this page

  function wireDropzone(inputId, labelId, onChange) {
    const input = $(inputId), label = $(labelId);
    input.addEventListener('change', () => {
      label.textContent = input.files[0] ? input.files[0].name : 'Drop image or click to browse';
      onChange();
    });
  }

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

  function setSpinner(ids, on) {
    ids.forEach((id) => { const el = $(id); if (el) el.classList.toggle('active', on); });
  }

  /* ================= FREQUENCY FILTERING PANEL ================= */
  const fqRun = makeDebouncedRunner(450);

  function fqSyncVisibility() {
    const ft = $('fq-ftype').value;
    $('fq-lowhigh').style.display = (ft === 'band_pass' || ft === 'band_stop') ? 'block' : 'none';
    $('fq-cutoffwrap').style.display = (ft === 'low_pass' || ft === 'high_pass') ? 'block' : 'none';
    $('fq-shiftwrap').style.display = (ft === 'brightness') ? 'block' : 'none';
  }

  function fqSyncCutoffRanges() {
    const n = parseInt($('fq-size').value, 10);
    [$('fq-rlow'), $('fq-rhigh'), $('fq-cutoff')].forEach((el) => {
      el.max = n;
      if (parseInt(el.value, 10) > n) el.value = n;
    });
    $('fq-rlowval').textContent = $('fq-rlow').value;
    $('fq-rhighval').textContent = $('fq-rhigh').value;
    $('fq-cutoffval').textContent = $('fq-cutoff').value;
  }

  async function fqProcess(signal) {
    if (!$('fq-file').files[0]) return;
    const n = $('fq-size').value;
    $('fq-status').textContent = `Processing at ${n}\u00d7${n}\u2026`;
    setSpinner(['fq-spin1', 'fq-spin2', 'fq-spin3', 'fq-spin4'], true);

    const fd = new FormData();
    fd.append('image', $('fq-file').files[0]);
    fd.append('filter_type', $('fq-ftype').value);
    fd.append('r_low', $('fq-rlow').value);
    fd.append('r_high', $('fq-rhigh').value);
    fd.append('cutoff', $('fq-cutoff').value);
    fd.append('shift_amount', $('fq-shift').value);
    fd.append('size', $('fq-size').value);

    try {
      const res = await fetch('/api/process', { method: 'POST', body: fd, signal });
      const data = await res.json();
      setSpinner(['fq-spin1', 'fq-spin2', 'fq-spin3', 'fq-spin4'], false);
      if (!res.ok) { $('fq-status').textContent = 'Error: ' + data.error; return; }
      $('fq-imgOrig').src = 'data:image/png;base64,' + data.original;
      $('fq-imgSpec').src = 'data:image/png;base64,' + data.spectrum;
      $('fq-imgFSpec').src = 'data:image/png;base64,' + data.filtered_spectrum;
      $('fq-imgRecon').src = 'data:image/png;base64,' + data.reconstructed;
      $('fq-status').textContent = 'Done.';

      const badge = $('fq-comp');
      if (data.complementarity) {
        badge.style.display = 'block';
        badge.className = 'freqBadge ' + (data.complementarity.is_valid ? 'pass' : 'fail');
        badge.textContent = (data.complementarity.is_valid ? 'Complementarity holds' : 'Complementarity failed')
          + ` \u00b7 max \u0394 = ${data.complementarity.delta.toExponential(2)}`;
      } else {
        badge.style.display = 'none';
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setSpinner(['fq-spin1', 'fq-spin2', 'fq-spin3', 'fq-spin4'], false);
        $('fq-status').textContent = 'Request failed: ' + err.message;
      }
      throw err;
    }
  }

  wireDropzone('fq-file', 'fq-filename', () => fqRun(fqProcess));
  $('fq-ftype').addEventListener('change', () => { fqSyncVisibility(); fqRun(fqProcess); });
  ['fq-rlow', 'fq-rhigh', 'fq-cutoff', 'fq-shift'].forEach((id) => {
    $(id).addEventListener('input', () => { $(id + 'val').textContent = $(id).value; fqRun(fqProcess); });
  });
  $('fq-size').addEventListener('input', () => {
    $('fq-sizeval').textContent = $('fq-size').value;
    fqSyncCutoffRanges();
    fqRun(fqProcess);
  });
  fqSyncVisibility();
  fqSyncCutoffRanges();

  /* ================= HYBRID IMAGE PANEL ================= */
  const hyRun = makeDebouncedRunner(450);

  function hySyncCutoffRanges() {
    const n = parseInt($('hy-size').value, 10);
    $('hy-cutoff').max = n;
    if (parseInt($('hy-cutoff').value, 10) > n) $('hy-cutoff').value = n;
    $('hy-cutoffval').textContent = $('hy-cutoff').value;
    $('hy-gainval').textContent = parseFloat($('hy-gain').value).toFixed(1) + '\u00d7';
    $('hy-sizeval').textContent = $('hy-size').value;
  }

  function applyFlipDisplay() {
    $('hy-flip-wrap').style.display = $('hy-flip').checked ? 'block' : 'none';
  }

  async function hyProcess(signal) {
    if (!$('hy-file-low').files[0] || !$('hy-file-high').files[0]) return;
    const n = $('hy-size').value;
    $('hy-status').textContent = `Building hybrid at ${n}\u00d7${n} (2\u00d7 forward + inverse CFT)\u2026`;
    setSpinner(['hy-spin'], true);

    const fd = new FormData();
    fd.append('image_low', $('hy-file-low').files[0]);
    fd.append('image_high', $('hy-file-high').files[0]);
    fd.append('cutoff', $('hy-cutoff').value);
    fd.append('high_gain', $('hy-gain').value);
    fd.append('size', $('hy-size').value);

    try {
      const res = await fetch('/api/hybrid', { method: 'POST', body: fd, signal });
      const data = await res.json();
      setSpinner(['hy-spin'], false);
      if (!res.ok) { $('hy-status').textContent = 'Error: ' + data.error; return; }
      $('hy-imgSourceLow').src = 'data:image/png;base64,' + data.source_low;
      $('hy-imgSourceHigh').src = 'data:image/png;base64,' + data.source_high;
      $('hy-imgLow').src = 'data:image/png;base64,' + data.low_component;
      $('hy-imgHigh').src = 'data:image/png;base64,' + data.high_component;
      $('hy-imgNear').src = 'data:image/png;base64,' + data.hybrid_near;
      $('hy-imgMid').src = 'data:image/png;base64,' + data.hybrid_mid;
      $('hy-imgFar').src = 'data:image/png;base64,' + data.hybrid_far;
      $('hy-imgFlipped').src = 'data:image/png;base64,' + data.hybrid_flipped;
      $('hy-status').textContent = `Done. (contrast auto-balance: ${data.auto_scale.toFixed(2)}\u00d7)`;

      const badge = $('hy-equiv');
      badge.style.display = 'block';
      badge.className = 'freqBadge pass';
      badge.textContent = `hybrid = IFFT(low_filter(FFT(a)) + ${data.auto_scale.toFixed(2)}\u00b7high_filter(FFT(b))) \u2014 verified vs. sum of separate inverses, \u0394 = ${data.equivalence_delta.toExponential(2)}`;
    } catch (err) {
      if (err.name !== 'AbortError') {
        setSpinner(['hy-spin'], false);
        $('hy-status').textContent = 'Request failed: ' + err.message;
      }
      throw err;
    }
  }

  wireDropzone('hy-file-low', 'hy-filename-low', () => hyRun(hyProcess));
  wireDropzone('hy-file-high', 'hy-filename-high', () => hyRun(hyProcess));
  $('hy-cutoff').addEventListener('input', () => { $('hy-cutoffval').textContent = $('hy-cutoff').value; hyRun(hyProcess); });
  $('hy-gain').addEventListener('input', () => { $('hy-gainval').textContent = parseFloat($('hy-gain').value).toFixed(1) + '\u00d7'; hyRun(hyProcess); });
  $('hy-size').addEventListener('input', () => { hySyncCutoffRanges(); hyRun(hyProcess); });
  $('hy-flip').addEventListener('change', applyFlipDisplay);
  hySyncCutoffRanges();
})();