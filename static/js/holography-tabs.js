// Tab switching between the Holography page's tools: Hologram retrieval (GS),
// Animated hologram, and Contour -> Hologram. Mirrors photography.js's rail
// pattern, just as a horizontal tab bar since there are only three tools so far.
(function () {
  const tabs = Array.from(document.querySelectorAll(".holoTab"));
  const panels = Array.from(document.querySelectorAll(".holoToolPanel"));
  if (!tabs.length) return;

  function activate(name) {
    tabs.forEach((t) => {
      const on = t.dataset.tool === name;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
    });
    panels.forEach((p) => p.classList.toggle("active", p.dataset.tool === name));
  }

  tabs.forEach((t) => t.addEventListener("click", () => activate(t.dataset.tool)));
  activate(tabs.find((t) => t.classList.contains("active"))?.dataset.tool || tabs[0].dataset.tool);
})();