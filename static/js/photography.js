(function () {
  'use strict';
  const btns = document.querySelectorAll('.railBtn');
  const panels = document.querySelectorAll('.hubPanel');

  function activate(name) {
    btns.forEach((b) => {
      const on = b.dataset.panel === name;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', String(on));
    });
    panels.forEach((p) => p.classList.toggle('active', p.dataset.panel === name));
  }

  btns.forEach((b) => b.addEventListener('click', () => activate(b.dataset.panel)));
  document.querySelectorAll('[data-goto]').forEach((el) => el.addEventListener('click', () => activate(el.dataset.goto)));
})();
