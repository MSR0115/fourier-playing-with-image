/* theme.js: applied synchronously in <head> so the page never flashes the wrong theme.
   'system' (no attribute) follows prefers-color-scheme; otherwise 'light' or 'dark' pins it. */
(function () {
  const KEY = 'fpi-theme';
  const root = document.documentElement;

  function apply(t) {
    if (t === 'light' || t === 'dark') root.setAttribute('data-theme', t);
    else root.removeAttribute('data-theme');
  }
  let saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) { /* private mode etc. */ }
  apply(saved);

  function current() {
    const explicit = root.getAttribute('data-theme');
    if (explicit) return explicit;
    return window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function set(t) {
    apply(t);
    try { localStorage.setItem(KEY, t); } catch (e) { /* ignore */ }
    document.dispatchEvent(new CustomEvent('fpi-themechange', { detail: { theme: t } }));
  }
  function toggle() { set(current() === 'dark' ? 'light' : 'dark'); }

  window.FPITheme = { current, set, toggle };
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.themeToggle').forEach((b) => {
      b.setAttribute('aria-label', 'Switch theme');
      b.addEventListener('click', toggle);
    });
  });
})();
