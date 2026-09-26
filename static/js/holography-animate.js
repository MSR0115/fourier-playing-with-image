// Animated hologram tool. POST /api/holo2d/anim/build returns a whole sequence
// of frames up front, each with its own converged 2-D hologram -- a target
// image, a hue-mapped phase image, and a reconstructed image. This file just
// plays that sequence back.
(function () {
  const panel = document.querySelector('.holoToolPanel[data-tool="anim"]');
  if (!panel) return;

  const kindBtns = Array.from(panel.querySelectorAll("[data-kind]"));
  const framesInput = panel.querySelector("#anim-frames");
  const itersInput = panel.querySelector("#anim-iters");
  const carryInput = panel.querySelector("#anim-carry");
  const buildBtn = panel.querySelector("#anim-build");
  const playBtn = panel.querySelector("#anim-play");
  const stepBtn = panel.querySelector("#anim-step");
  const speedInput = panel.querySelector("#anim-speed");
  const frameLabel = panel.querySelector("#anim-frameLabel");
  const rmseLabel = panel.querySelector("#anim-rmse");
  const filmstrip = panel.querySelector("#anim-filmstrip");
  const status = panel.querySelector("#anim-status");
  const targetImg = panel.querySelector("#anim-targetImg");
  const phaseImg = panel.querySelector("#anim-phaseImg");
  const reconImg = panel.querySelector("#anim-reconImg");
  const sparkCanvas = panel.querySelector("#anim-spark");

  let kind = "moving_dot";
  let frames = [];
  let idx = 0;
  let timer = null;

  kindBtns.forEach((b) => {
    b.addEventListener("click", () => {
      kind = b.dataset.kind;
      kindBtns.forEach((x) => x.classList.toggle("active", x === b));
    });
  });

  function setStatus(msg) {
    if (status) status.textContent = msg;
  }

  async function build() {
    stop();
    setStatus("Drawing frames and converging a hologram for each\u2026");
    buildBtn.disabled = true;
    try {
      const res = await fetch("/api/holo2d/anim/build", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kind,
          frames: Number(framesInput.value),
          iterations: Number(itersInput.value),
          carryPhase: !!carryInput.checked,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Build failed.");
      frames = data.frames;
      idx = 0;
      renderFilmstrip();
      renderFrame();
      setStatus(`Built ${frames.length} frames (\u201c${kind.replace("_", " ")}\u201d).`);
    } catch (err) {
      setStatus(err.message);
    } finally {
      buildBtn.disabled = false;
    }
  }

  function renderFilmstrip() {
    filmstrip.innerHTML = "";
    frames.forEach((f, i) => {
      const im = document.createElement("img");
      im.src = f.reconImg;
      im.className = "animFrameThumb" + (i === idx ? " current" : "");
      im.alt = `Frame ${i + 1}`;
      im.addEventListener("click", () => {
        stop();
        idx = i;
        renderFrame();
      });
      filmstrip.appendChild(im);
    });
  }

  function markFilmstrip() {
    Array.from(filmstrip.children).forEach((c, i) => c.classList.toggle("current", i === idx));
  }

  function renderFrame() {
    if (!frames.length) return;
    const f = frames[idx];
    frameLabel.textContent = `Frame ${idx + 1} / ${frames.length}`;
    rmseLabel.textContent = f.rmse.toFixed(4);
    targetImg.src = f.targetImg;
    phaseImg.src = f.phaseImg;
    reconImg.src = f.reconImg;
    drawSpark();
    markFilmstrip();
  }

  function drawSpark() {
    const ctx = sparkCanvas.getContext("2d");
    const w = sparkCanvas.width, h = sparkCanvas.height;
    ctx.clearRect(0, 0, w, h);
    if (!frames.length) return;
    const vals = frames.map((f) => f.rmse);
    const maxV = Math.max(1e-6, ...vals);
    ctx.beginPath();
    vals.forEach((v, i) => {
      const x = (i / (vals.length - 1 || 1)) * (w - 6) + 3;
      const y = h - 4 - (v / maxV) * (h - 8);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "rgba(106,166,255,0.85)";
    ctx.lineWidth = 2;
    ctx.stroke();
    const cx = (idx / (vals.length - 1 || 1)) * (w - 6) + 3;
    const cy = h - 4 - (vals[idx] / maxV) * (h - 8);
    ctx.beginPath();
    ctx.arc(cx, cy, 3.5, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(255,159,106,0.95)";
    ctx.fill();
  }

  function step(dir = 1) {
    if (!frames.length) return;
    idx = (idx + dir + frames.length) % frames.length;
    renderFrame();
  }

  function play() {
    if (!frames.length || timer) return;
    playBtn.textContent = "Pause";
    const speed = () => 900 - Number(speedInput.value) * 80;
    const tick = () => {
      step(1);
      timer = setTimeout(tick, speed());
    };
    timer = setTimeout(tick, speed());
  }

  function stop() {
    if (timer) clearTimeout(timer);
    timer = null;
    playBtn.textContent = "Play";
  }

  buildBtn.addEventListener("click", build);
  stepBtn.addEventListener("click", () => {
    stop();
    step(1);
  });
  playBtn.addEventListener("click", () => (timer ? stop() : play()));

  sparkCanvas.width = 560;
  sparkCanvas.height = 90;

  build();
})();