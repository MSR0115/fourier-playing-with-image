/* landing.js: the hero demo. A handful of rotating circles (the same idea the
   Epicycle Drawing tool runs on a face) tracing a small closed curve, forever. */
(function () {
  'use strict';
  const cv = document.getElementById('demo');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const DPR = Math.min(window.devicePixelRatio || 1, 2);

  function fit() {
    const s = cv.clientWidth || 148;
    cv.width = Math.round(s * DPR);
    cv.height = Math.round(s * DPR);
  }
  fit();
  window.addEventListener('resize', fit);

  /* four circles is enough to make a simple ring feel like a flower */
  const CIRCLES = [{ f: 1, r: 0.50 }, { f: -3, r: 0.20 }, { f: 5, r: 0.11 }, { f: -7, r: 0.05 }];
  const N = 240;
  const path = new Float32Array(N * 2);
  for (let i = 0; i < N; i++) {
    const t = i / N;
    let x = 0, y = 0;
    for (const c of CIRCLES) { const a = 2 * Math.PI * c.f * t; x += c.r * Math.cos(a); y += c.r * Math.sin(a); }
    path[i * 2] = x; path[i * 2 + 1] = y;
  }
  function pointAt(t) {
    let x = 0, y = 0;
    for (const c of CIRCLES) { const a = 2 * Math.PI * c.f * t; x += c.r * Math.cos(a); y += c.r * Math.sin(a); }
    return [x, y];
  }

  function colors() {
    const cs = getComputedStyle(document.documentElement);
    return { ink: cs.getPropertyValue('--ink').trim(), accent: cs.getPropertyValue('--accent').trim() };
  }
  let col = colors();
  document.addEventListener('fpi-themechange', () => { col = colors(); });

  const reduceMotion = !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);
  const PERIOD = 7000;
  const t0 = performance.now();

  function frame(now) {
    requestAnimationFrame(frame);
    const w = cv.width, h = cv.height, cx = w / 2, cy = h / 2, s = w * 0.42;
    const t = reduceMotion ? 0.16 : ((now - t0) % PERIOD) / PERIOD;
    ctx.clearRect(0, 0, w, h);

    ctx.beginPath();
    for (let i = 0; i <= N; i++) {
      const idx = (i % N) * 2, x = cx + path[idx] * s, y = cy + path[idx + 1] * s;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = col.ink; ctx.globalAlpha = 0.14; ctx.lineWidth = 1.3 * DPR; ctx.stroke(); ctx.globalAlpha = 1;

    let x = cx, y = cy;
    for (const c of CIRCLES) {
      const rad = c.r * s;
      ctx.beginPath(); ctx.arc(x, y, rad, 0, 2 * Math.PI);
      ctx.strokeStyle = col.accent; ctx.globalAlpha = 0.32; ctx.lineWidth = 1 * DPR; ctx.stroke(); ctx.globalAlpha = 1;
      const a = 2 * Math.PI * c.f * t, nx = x + rad * Math.cos(a), ny = y + rad * Math.sin(a);
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(nx, ny);
      ctx.strokeStyle = col.accent; ctx.globalAlpha = 0.8; ctx.lineWidth = 1.1 * DPR; ctx.stroke(); ctx.globalAlpha = 1;
      x = nx; y = ny;
    }
    ctx.beginPath(); ctx.arc(x, y, 2.6 * DPR, 0, 2 * Math.PI); ctx.fillStyle = col.accent; ctx.fill();

    if (!reduceMotion) {
      const tail = 40;
      ctx.beginPath();
      for (let k = tail; k >= 0; k--) {
        const tn = (((t - k / N) % 1) + 1) % 1, [px, py] = pointAt(tn), sx = cx + px * s, sy = cy + py * s;
        if (k === tail) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
      }
      ctx.strokeStyle = col.accent; ctx.lineWidth = 2 * DPR; ctx.lineCap = 'round'; ctx.globalAlpha = 0.9; ctx.stroke(); ctx.globalAlpha = 1;
    }
  }
  requestAnimationFrame(frame);
})();
