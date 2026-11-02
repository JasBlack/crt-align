// ─────────────────────────────────────────────────────────────────────────────
// JAS.BLACK tool family — shared behaviour. Load after the page markup:
//   <script src="family.js"></script>   (or paste it inline in a single-file build)
// Set the version once: <body data-version="0.1.0">. The tool's own code can call
// JB.toast('Saved') and JB.paintRange(input) after it sets a slider from code.
// ─────────────────────────────────────────────────────────────────────────────
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const store = { get: k => { try { return localStorage.getItem(k); } catch (e) { return null; } },
                  set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* private window */ } } };

  // version: shown only while the title is hovered (.brand-ver right after .brand)
  const ver = document.body.dataset.version;
  for (const el of $$('.brand-ver')) if (ver && !el.textContent) el.textContent = 'v' + ver;

  // light / dark, remembered per tool (key: <tool>-ui). The stage stays dark either way.
  const tool = (document.body.dataset.tool || location.hostname.split('.')[0] || 'tool').toLowerCase();
  const setUI = t => { document.body.dataset.ui = t; store.set(tool + '-ui', t); };
  setUI(store.get(tool + '-ui') === 'light' ? 'light' : (document.body.dataset.ui || 'dark'));
  for (const b of $$('[data-action=theme]')) b.addEventListener('click', () => setUI(document.body.dataset.ui === 'dark' ? 'light' : 'dark'));

  // sliders: green fill up to the value
  const paintRange = el => el.style.setProperty('--pct', ((el.value - el.min) / ((el.max - el.min) || 1) * 100) + '%');
  document.addEventListener('input', e => { if (e.target.type === 'range') paintRange(e.target); });

  // right-click any slider: back to its default (its value once the page has loaded).
  // Family rule — every tool does this. A slider that needs its own right-click keeps it
  // by calling preventDefault() first (e.g. Growth: Shift+right-click = MIDI learn).
  const defs = new WeakMap();
  const remember = () => { for (const el of $$('input[type=range]')) { if (!defs.has(el)) defs.set(el, el.value); paintRange(el); } };
  if (document.readyState === 'complete') remember(); else addEventListener('load', remember);
  new MutationObserver(remember).observe(document.body, { childList: true, subtree: true }); // sliders built later
  document.addEventListener('contextmenu', e => {
    const el = e.target;
    if (e.defaultPrevented || !(el instanceof HTMLInputElement) || el.type !== 'range' || el.disabled) return;
    e.preventDefault();
    const d = defs.get(el) ?? el.defaultValue;
    if (el.value === d) return;
    el.value = d;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  });

  // segmented switches: one pressed at a time
  for (const seg of $$('.seg')) seg.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    for (const o of $$('button', seg)) o.setAttribute('aria-pressed', String(o === b));
  });

  // full screen key
  for (const b of $$('[data-action=fullscreen]')) b.addEventListener('click', () => {
    const t = $(b.dataset.target || '.monitor') || document.documentElement;
    document.fullscreenElement ? document.exitFullscreen?.() : t.requestFullscreen?.();
  });

  // toast, bottom right
  let toastTimer = 0;
  const toast = msg => {
    let t = $('#toast'); if (!t) { t = document.createElement('div'); t.id = 'toast'; t.setAttribute('role', 'status'); document.body.append(t); }
    t.textContent = msg; t.classList.add('visible'); clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('visible'), 2600);
  };

  window.JB = { toast, paintRange, setUI };
})();
