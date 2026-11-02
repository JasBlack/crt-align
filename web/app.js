import { Renderer } from './gl/renderer.js';
import { loadImage, loadVideo, lumaAt, lumaSignature, saveFile, prepareDownloads, isVideoFile, mb } from './media.js';
import * as engine from './engine.js';
import {
  modelToJSON, modelFromJSON, setScreen, makeScreen, screenEval, contentST,
} from './core/model.js';
import { detectScreen } from './core/screen.js';
import { syncCandidates } from './core/sync.js';
import { lumaFromRGBA } from './core/image.js';
import { HandWarp } from './hand.js';

const $ = id => document.getElementById(id);

// ---------------------------------------------------------------- theme
// colours for canvas drawing come from the same CSS tokens as the page
let C = {};
const rgb01 = v => {
  v = v.trim();
  const h = v.length === 4 ? v.slice(1).split('').map(c => c + c).join('') : v.slice(1);
  return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16) / 255);
};
function readTheme() {
  const cs = getComputedStyle(document.body), g = n => cs.getPropertyValue(n).trim();
  C = { accent: g('--accent'), accent2: g('--accent2'), panel: g('--panel'), well: rgb01(g('--well')),
        line: rgb01(g('--accent')), light: document.body.dataset.ui === 'light' };
}
function setTheme(ui) {
  document.body.dataset.ui = ui;
  try { localStorage.setItem('crt-align-ui', ui); } catch { /* storage blocked */ }
  readTheme(); draw();
}
try { document.body.dataset.ui = localStorage.getItem('crt-align-ui') || localStorage.getItem('crt-match-ui') || 'dark'; } catch { /* storage blocked */ }
readTheme();

const S = {
  hand: new HandWarp(), sel: -1, keyT: null,
  orig: null, film: null, model: null, score: null,
  mode: 1, opacity: 0.5, swipe: 0.5, tool: 'view', edge: 0,
  quality: 'medium', lambda: 0.3,
  view: null, rawView: null,
  offset: 0, t: 0, playing: false,
  busy: false, cancel: false,
  history: [], future: [],
  follow: false, refDetect: null, tracked: null,
  rasterEdited: false,
};

// ---------------------------------------------------------------- canvases

const stage = $('stage'), glCanvas = $('gl'), ov = $('overlay');
let R;
try {
  // preserved buffer: screenshots and thumbnails capture what's on screen
  R = new Renderer(glCanvas, { preserve: true });
} catch (e) {
  $('stage-hint').hidden = false;
  $('stage-hint').textContent = e.message;
  throw e;
}
const octx = ov.getContext('2d');
let dpr = 1, VW = 1, VH = 1;
let xr = null; // export renderer, created on demand
const exportRenderer = () => (xr ??= new Renderer(document.createElement('canvas'), { preserve: true, alpha: true }));

function resize() {
  const r = stage.getBoundingClientRect();
  dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  const first = VW <= 1;
  VW = glCanvas.width = ov.width = w;
  VH = glCanvas.height = ov.height = h;
  if (first || !S.view) fitView();
  draw();
}
new ResizeObserver(resize).observe(stage);

function canvasDims() {
  if (S.tool === 'raster' || !S.orig) return S.film ? [S.film.w, S.film.h] : [16, 9];
  return [S.orig.w, S.orig.h];
}
const rawMode = () => S.tool === 'raster' || !S.orig;
const curView = () => (rawMode() ? S.rawView : S.view);

function fitView(which = null) {
  const fitOne = raw => {
    const [w, h] = raw ? (S.film ? [S.film.w, S.film.h] : [16, 9]) : (S.orig ? [S.orig.w, S.orig.h] : [16, 9]);
    const z = Math.min(VW / w, VH / h) * 0.94;
    return { x: (VW - w * z) / 2, y: (VH - h * z) / 2, z };
  };
  if (which !== 'raw') S.view = fitOne(!S.orig);
  if (which !== 'warp') S.rawView = fitOne(true);
  updateZoom();
}

const evPos = e => { const r = stage.getBoundingClientRect(); return [(e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr]; };
const toCanvas = ([bx, by], v = curView()) => [(bx - v.x) / v.z, (by - v.y) / v.z];
const toBuf = (cx, cy, v = curView()) => [cx * v.z + v.x, cy * v.z + v.y];

// ---------------------------------------------------------------- drawing

let raf = 0;
function draw() { if (!raf) raf = requestAnimationFrame(() => { raf = 0; paint(); }); }

function paint() {
  const view = curView();
  if (!view) return;
  const [cw, ch] = canvasDims();
  let mode = S.mode;
  if (rawMode()) mode = 7;
  else if (!S.model || !S.film) mode = 6;
  const tile = Math.max(16, Math.round(Math.min(cw, ch) / 12));
  R.render({ mode, opacity: S.opacity, tile, swipe: S.swipe, view, canvasSize: [cw, ch], width: VW, height: VH,
             edge: S.edge, screenOverride: S.follow ? S.tracked : null,
             bg: C.well, dots: C.light ? [0, 0, 0, 0.9] : [1, 1, 1, 1], dpr, line: C.line });
  paintOverlay();
}

function paintOverlay() {
  octx.clearRect(0, 0, VW, VH);
  if (!S.model) return;
  if (S.tool === 'raster') return paintRaster();
  const v = S.view, W = S.orig.w, H = S.orig.h;
  octx.lineWidth = 1.5 * dpr;
  if (S.tool === 'pin' || S.hand.count) paintHand(v, W, H);
  if (S.mode === 4 && S.tool === 'view') {
    const [sx] = toBuf(S.swipe * W, 0, v), [, sy0] = toBuf(0, 0, v), [, sy1] = toBuf(0, H, v);
    const y = Math.max(sy0, Math.min(sy1, VH / 2));
    const g = octx.createLinearGradient(sx - 8 * dpr, y - 8 * dpr, sx + 8 * dpr, y + 8 * dpr);
    g.addColorStop(0, '#f4f5f7'); g.addColorStop(1, '#c9ccd4');
    octx.shadowColor = 'rgba(0,0,0,0.55)'; octx.shadowBlur = 6 * dpr; octx.shadowOffsetY = 2 * dpr;
    octx.fillStyle = g;
    octx.beginPath(); octx.arc(sx, y, 9 * dpr, 0, Math.PI * 2); octx.fill();
    octx.shadowColor = 'transparent';
  }
}

// puppet pins: purple dots, the dragged/selected one filled; grid pins joined by lines
function paintHand(v, W, H) {
  const pins = S.hand.pins, at = p => toBuf(p.pos[0] * W, p.pos[1] * H, v);
  const dim = S.tool !== 'pin';
  octx.globalAlpha = dim ? 0.45 : 1;
  if (S.hand.hasGrid) {
    const [gx, gy] = S.hand.gridSize || [4, 3];
    const node = (i, j) => {
      if ((i === 0 || i === gx) && (j === 0 || j === gy)) return [i / gx, j / gy];
      const p = pins.find(q => q.grid && q.grid[0] === i && q.grid[1] === j);
      return p ? p.pos : null;
    };
    octx.strokeStyle = C.accent2; octx.lineWidth = 1 * dpr; octx.globalAlpha *= 0.55;
    octx.beginPath();
    for (let j = 0; j <= gy; j++) for (let i = 0; i <= gx; i++) {
      const a = node(i, j); if (!a) continue;
      for (const [di, dj] of [[1, 0], [0, 1]]) {
        const b = node(i + di, j + dj); if (!b) continue;
        const [x0, y0] = toBuf(a[0] * W, a[1] * H, v), [x1, y1] = toBuf(b[0] * W, b[1] * H, v);
        octx.moveTo(x0, y0); octx.lineTo(x1, y1);
      }
    }
    octx.stroke();
    octx.globalAlpha = dim ? 0.45 : 1;
  }
  pins.forEach((p, i) => {
    const [x, y] = at(p), r = (p.grid ? 4.5 : 6) * dpr, on = i === S.sel && !dim;
    octx.fillStyle = 'rgba(0,0,0,0.55)';
    octx.beginPath(); octx.arc(x, y, r + 2.2 * dpr, 0, Math.PI * 2); octx.fill();
    octx.fillStyle = on ? C.accent2 : '#f4f5f7';
    octx.beginPath(); octx.arc(x, y, r, 0, Math.PI * 2); octx.fill();
    octx.fillStyle = on ? '#f4f5f7' : C.accent2;
    octx.beginPath(); octx.arc(x, y, r * 0.42, 0, Math.PI * 2); octx.fill();
  });
  octx.globalAlpha = 1;
}

function ring(x, y, r, col) {
  octx.strokeStyle = 'rgba(0,0,0,0.6)'; octx.lineWidth = 3.5 * dpr;
  octx.beginPath(); octx.arc(x, y, r, 0, Math.PI * 2); octx.stroke();
  octx.strokeStyle = col; octx.lineWidth = 1.6 * dpr;
  octx.beginPath(); octx.arc(x, y, r, 0, Math.PI * 2); octx.stroke();
  octx.beginPath(); octx.moveTo(x - r * 1.6, y); octx.lineTo(x - r * 0.5, y); octx.moveTo(x + r * 0.5, y); octx.lineTo(x + r * 1.6, y);
  octx.moveTo(x, y - r * 1.6); octx.lineTo(x, y - r * 0.5); octx.moveTo(x, y + r * 0.5); octx.lineTo(x, y + r * 1.6); octx.stroke();
}


const _sc = new Float64Array(6);
function screenPt(scr, s, t) { screenEval(scr, s, t, _sc); return [_sc[0], _sc[1]]; }
const EDGE_MID = [[0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]];

function rasterHandles() {
  const scr = S.model.screen;
  const corners = scr.corners.map((c, i) => ({ kind: 'corner', i, p: c }));
  const mids = EDGE_MID.map(([s, t], i) => ({ kind: 'bulge', i, p: screenPt(scr, s, t) }));
  return corners.concat(mids);
}

function paintRaster() {
  const scr = S.model.screen, v = S.rawView, ins = S.model.inset;
  const outline = (a, b) => {
    octx.beginPath();
    const seq = [[0, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 0, 0]];
    let first = true;
    for (const [s0, t0, s1, t1] of seq) for (let k = 0; k <= 32; k++) {
      const f = k / 32;
      const s = a + (b - a) * (s0 + (s1 - s0) * f), t = a + (b - a) * (t0 + (t1 - t0) * f);
      const [x, y] = toBuf(...screenPt(scr, s, t), v);
      if (first) { octx.moveTo(x, y); first = false; } else octx.lineTo(x, y);
    }
    octx.closePath(); octx.stroke();
  };
  octx.lineWidth = 1.5 * dpr; octx.strokeStyle = C.accent; octx.setLineDash([]);
  outline(0, 1);
  octx.setLineDash([6 * dpr, 5 * dpr]); octx.globalAlpha = 0.75;
  outline(ins, 1 - ins);
  octx.setLineDash([]); octx.globalAlpha = 1;
  for (const h of rasterHandles()) {
    const [x, y] = toBuf(h.p[0], h.p[1], v), r = 6 * dpr;
    octx.fillStyle = drag && drag.handle && drag.handle.kind === h.kind && drag.handle.i === h.i ? C.accent : C.panel;
    octx.strokeStyle = C.accent; octx.lineWidth = 2 * dpr;
    octx.beginPath();
    if (h.kind === 'corner') octx.rect(x - r, y - r, 2 * r, 2 * r); else octx.arc(x, y, r, 0, Math.PI * 2);
    octx.fill(); octx.stroke();
  }
}

// ---------------------------------------------------------------- interaction

let drag = null, spaceDown = false;

stage.addEventListener('wheel', e => {
  e.preventDefault();
  const v = curView(); if (!v) return;
  const [bx, by] = evPos(e);
  const k = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015));
  const z = Math.max(0.02, Math.min(40, v.z * k)), f = z / v.z;
  v.x = bx - (bx - v.x) * f; v.y = by - (by - v.y) * f; v.z = z;
  updateZoom(); draw();
}, { passive: false });

stage.addEventListener('contextmenu', e => e.preventDefault());
stage.addEventListener('pointerdown', e => {
  if (S.busy) return;
  stage.setPointerCapture(e.pointerId);
  const pos = evPos(e), c = toCanvas(pos);
  const panButton = e.button === 1 || e.button === 2 || spaceDown;
  if (!panButton && e.button === 0 && S.model) {
    if (S.tool === 'pin' && !rawMode()) {
      const W = S.orig.w, H = S.orig.h, u = c[0] / W, v = c[1] / H;
      let i = S.hand.nearest(u, v, (11 * dpr) / (S.view.z * H));
      if (e.altKey) {
        if (i >= 0) { pushHistory(); S.hand.remove(i); S.sel = -1; applyHand(); handEdited(); }
        return;
      }
      pushHistory();
      if (i < 0) i = S.hand.add(u, v);
      if (i < 0) { toast(`That's the limit of ${S.hand.count} pins. Remove some, or clear them.`, true); S.history.pop(); return; }
      const p = S.hand.pins[i];
      S.sel = i;
      drag = { type: 'hand', i, grab: [u - p.pos[0], v - p.pos[1]] };
      applyHand();
      return;
    }
    if (S.tool === 'raster') {
      const hit = rasterHandles().find(h => { const [x, y] = toBuf(h.p[0], h.p[1]); return Math.hypot(x - pos[0], y - pos[1]) < 12 * dpr; });
      if (hit) { pushHistory(); drag = { type: 'handle', handle: hit }; draw(); return; }
    }
    if (S.tool === 'view' && S.mode === 4 && !rawMode()) { drag = { type: 'swipe' }; moveSwipe(c); return; }
  }
  drag = { type: 'pan', start: pos, v0: { ...curView() } };
  stage.classList.add('panning');
});
stage.addEventListener('pointermove', e => {
  if (!drag) return;
  const pos = evPos(e), c = toCanvas(pos);
  if (drag.type === 'pan') {
    const v = curView();
    v.x = drag.v0.x + pos[0] - drag.start[0]; v.y = drag.v0.y + pos[1] - drag.start[1];
  } else if (drag.type === 'hand') {
    S.hand.move(drag.i, c[0] / S.orig.w - drag.grab[0], c[1] / S.orig.h - drag.grab[1]);
    R.setHand(S.hand.sol);
  }
  else if (drag.type === 'swipe') moveSwipe(c);
  else if (drag.type === 'handle') moveHandle(drag.handle, c);
  draw();
});
const endDrag = async () => {
  if (!drag) return;
  const d = drag; drag = null;
  stage.classList.remove('panning');
  if (d.type === 'hand') {
    applyHand(); handEdited();
  } else if (d.type === 'handle') {
    S.rasterEdited = true;
    $('btn-refine').classList.add('nudge');
    lcd('screen edited. the layer already follows it; refine re-fits the auto mesh to the new edges.');
    markKey();
    draw();
  }
};
stage.addEventListener('pointerup', endDrag);
stage.addEventListener('pointercancel', endDrag);
stage.addEventListener('dblclick', () => {
  if (S.tool === 'pin') return; // double-clicks drop pins in the pin tool
  fitView(rawMode() ? 'raw' : 'warp'); draw();
});

function applyHand() { R.setHand(S.hand.sol); renderHand(); draw(); }
function handEdited() {
  markKey();
  const n = S.hand.count, m = S.hand.moved;
  lcd(n ? `hand warp: ${n} pin${n === 1 ? '' : 's'}, ${m} moved. it holds on every frame and in the export.` : 'hand warp cleared.');
  refreshAll();
}
// the frame a video's alignment / hand warp was set on
function markKey() { if (timeline()) { S.keyT = S.t; updateKeyMark(); } }
function updateKeyMark() {
  const tl = timeline(), mark = $('keymark');
  const show = !!(tl && S.keyT != null && S.model);
  mark.hidden = !show; $('key-row').hidden = !show;
  if (!show) return;
  // follow the thumb's travel: .tl padding (10px) + half the 14px thumb on each side
  const f = Math.max(0, Math.min(1, (S.keyT - tl.t0) / Math.max(1e-6, tl.t1 - tl.t0)));
  mark.style.left = `calc(17px + (100% - 34px) * ${f.toFixed(5)})`;
  $('key-val').textContent = `key frame ${timecodeOf(S.keyT, tl.fps)}`;
}

function moveSwipe(c) { S.swipe = Math.max(0, Math.min(1, c[0] / S.orig.w)); }

function moveHandle(h, c) {
  const scr = S.model.screen;
  const corners = scr.corners.map(p => p.slice()), bulge = scr.bulge.slice();
  if (h.kind === 'corner') {
    corners[h.i] = [c[0], c[1]];
  } else {
    // bulge = offset of the edge midpoint from the straight (homography) edge, along its outward normal
    const flat = makeScreen(corners, [0, 0, 0, 0]);
    const [s, t] = EDGE_MID[h.i], m = screenPt(flat, s, t), n = flat.N[h.i];
    bulge[h.i] = (c[0] - m[0]) * n[0] + (c[1] - m[1]) * n[1];
  }
  setScreen(S.model, corners, bulge);
  R.setModel(S.model);
}


window.addEventListener('keydown', e => {
  if (e.target.matches('input, textarea')) return;
  if (e.code === 'Space') { spaceDown = true; e.preventDefault(); return; }
  const k = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && k === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
  if ((e.ctrlKey || e.metaKey) && k === 'y') { e.preventDefault(); redo(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (k >= '1' && k <= '7') setMode(Number(k) - 1);
  else if (k === 'v') setTool('view');
  else if (k === 'p') setTool('pin');
  else if (k === 'r') setTool('raster');
  else if (k === 'f') { fitView(rawMode() ? 'raw' : 'warp'); draw(); }
  else if ((k === 'delete' || k === 'backspace') && S.tool === 'pin' && S.sel >= 0) {
    pushHistory(); S.hand.remove(S.sel); S.sel = -1; applyHand(); handEdited(); e.preventDefault();
  }
  else if (k === 'm' && S.model) toggleGrid();
  else if (k === 'escape') { S.sel = -1; draw(); }
  else if ((k === 'arrowleft' || k === 'arrowright') && timeline()) { stepFrame(k === 'arrowleft' ? -1 : 1); e.preventDefault(); }
});
window.addEventListener('keyup', e => { if (e.code === 'Space') spaceDown = false; });

// ---------------------------------------------------------------- controls

function pressed(group, attr, val) {
  for (const b of $(group).querySelectorAll('button')) b.setAttribute('aria-pressed', String(b.dataset[attr] === String(val)));
}
const TOOL_NOTES = {
  view: 'move: drag to pan, scroll to zoom, double-click to fit. space-drag pans with any tool.',
  pin: 'pin: click the glitch layer to drop a pin, drag it to warp live. pins hold their spot. alt-click removes one, delete removes the selected.',
  raster: 'screen: drag the corners and edge bulges onto the lit screen in the photo, then refine.',
};
function setMode(m) { S.mode = m; pressed('modes', 'mode', m); $('mix-wrap').hidden = m !== 1; draw(); }
function setTool(t) {
  if (t !== 'view' && !S.model) { toast('Align first, then use this tool.'); return; }
  S.tool = t; stage.dataset.tool = t; pressed('tools', 'tool', t);
  $('tool-note').textContent = TOOL_NOTES[t];
  updateZoom(); renderHint(); draw();
}
$('modes').addEventListener('click', e => { const b = e.target.closest('button'); if (b) setMode(Number(b.dataset.mode)); });
$('tools').addEventListener('click', e => { const b = e.target.closest('button'); if (b) setTool(b.dataset.tool); });
$('opacity').addEventListener('input', e => { S.opacity = Number(e.target.value); draw(); });
$('fit').addEventListener('click', () => { fitView(rawMode() ? 'raw' : 'warp'); draw(); });
$('quality').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { S.quality = b.dataset.q; pressed('quality', 'q', S.quality); } });
$('stiffness').addEventListener('input', e => { S.lambda = Math.pow(10, Number(e.target.value)); });
$('inset').addEventListener('input', e => {
  if (!S.model) return;
  S.model.inset = Number(e.target.value);
  $('inset-val').textContent = `${(S.model.inset * 100).toFixed(1)}%`;
  S.rasterEdited = true; R.setModel(S.model); draw();
});
$('inset').addEventListener('change', () => { if (S.model) { $('btn-refine').classList.add('nudge'); } });
$('edge-mode').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  pushHistory(); S.edge = Number(b.dataset.edge); pressed('edge-mode', 'edge', S.edge); updateBandNote(); updateOutLine(); draw();
});
$('btn-align').addEventListener('click', () => runAlign('align'));
$('btn-refine').addEventListener('click', () => runAlign('refine'));
$('btn-reset-mesh').addEventListener('click', () => {
  if (!S.model) return;
  pushHistory();
  S.model.cx.fill(0); S.model.cy.fill(0); S.model.pins = [];
  R.setModel(S.model); renderHand(); draw();
  lcd('auto mesh cleared. the screen and hand warp stay. refine to fit the mesh again.');
});
$('btn-clear-pins').addEventListener('click', () => { if (!S.hand.count) return; pushHistory(); S.hand.clear(); S.sel = -1; applyHand(); handEdited(); });
$('btn-grid').addEventListener('click', toggleGrid);
$('reach').addEventListener('input', e => {
  S.hand.setReach(Number(e.target.value));
  $('reach-val').textContent = `${Math.round(S.hand.reach * 100)}%`;
  R.setHand(S.hand.sol); draw();
});
$('reach').addEventListener('pointerdown', () => pushHistory());
$('btn-goto-key').addEventListener('click', () => {
  if (S.keyT == null) return;
  S.t = S.keyT; $('scrub').value = String(S.t); paintFader($('scrub')); updateTimecode(); loadFrames();
});
function toggleGrid() {
  if (!S.model) return;
  pushHistory();
  if (S.hand.hasGrid) S.hand.removeGrid(); else S.hand.addGrid(4, 3);
  if (S.tool !== 'pin') setTool('pin');
  S.sel = -1; applyHand(); handEdited();
}
$('undo').addEventListener('click', undo);
$('redo').addEventListener('click', redo);
$('btn-export').addEventListener('click', exportLayer);
$('btn-save-json').addEventListener('click', saveAlignment);
$('btn-load-json').addEventListener('click', () => $('json-input').click());
$('json-input').addEventListener('change', async e => { const f = e.target.files[0]; e.target.value = ''; if (f) await loadAlignment(f); });
$('follow').addEventListener('click', () => {
  S.follow = !S.follow; $('follow').setAttribute('aria-pressed', String(S.follow)); S.tracked = null; loadFrames();
});
$('theme').addEventListener('click', () => setTheme(document.body.dataset.ui === 'light' ? 'dark' : 'light'));
$('btn-sample').addEventListener('click', () => { if (!S.busy) loadSample(true); });
function paintFader(el) {
  const min = Number(el.min), max = Number(el.max), v = Number(el.value);
  el.style.setProperty('--pct', `${((v - min) / (max - min || 1)) * 100}%`);
}
for (const el of document.querySelectorAll('input.fad, #scrub')) { el.addEventListener('input', () => paintFader(el)); paintFader(el); }
$('off-minus').addEventListener('click', () => nudgeOffset(-1));
$('off-plus').addEventListener('click', () => nudgeOffset(1));
$('btn-sync').addEventListener('click', autoSync);
$('busy-cancel').addEventListener('click', () => { S.cancel = true; engine.cancelEngine(); });

function updateZoom() { const v = curView(); $('zoom-val').textContent = v ? `${Math.round(v.z * 100 / dpr)}%` : ''; }

function renderHand() {
  const n = S.hand.count;
  $('hand-count').textContent = n ? `${n} pin${n === 1 ? '' : 's'} \u00b7 ${S.hand.moved} moved` : 'no pins';
  $('btn-clear-pins').hidden = n === 0;
  $('btn-grid').setAttribute('aria-pressed', String(S.hand.hasGrid));
  $('btn-grid').disabled = !S.model;
  $('reach').value = String(S.hand.reach); $('reach-val').textContent = `${Math.round(S.hand.reach * 100)}%`;
  paintFader($('reach'));
}

const VU_N = 14;
$('vu').innerHTML = '<i></i>'.repeat(VU_N);
function updateScore() {
  const s = S.score;
  $('score-val').textContent = s == null ? '\u2013' : `${Math.round(s * 100)}%`;
  // full scale at 70%: real CRT shots land around 35-60%
  const lit = s == null ? 0 : Math.round(Math.max(0, Math.min(1, s / 0.7)) * VU_N);
  const cls = s == null ? '' : s >= 0.4 ? 'g' : s >= 0.25 ? 'p' : 'r';
  [...$('vu').children].forEach((el, i) => { el.className = i < lit ? cls : ''; });
}

// how much of the original's frame the screen cut off, per axis
function overscan() {
  const m = S.model, p = [0, 0];
  let du = 0, dv = 0;
  for (let k = 0; k <= 20; k++) {
    const q = k / 20;
    contentST(m, 0, q, p); du = Math.max(du, -p[0]); contentST(m, 1, q, p); du = Math.max(du, p[0] - 1);
    contentST(m, q, 0, p); dv = Math.max(dv, -p[1]); contentST(m, q, 1, p); dv = Math.max(dv, p[1] - 1);
  }
  return [Math.max(0, du), Math.max(0, dv)];
}

function updateBandNote() {
  const el = $('band-note');
  if (!S.model) { el.textContent = 'pin corners fills the frame edge to edge. exact leaves what the screen never showed transparent.'; return; }
  const [ou, ov] = overscan();
  const cut = Math.max(ou, ov) > 0.004
    ? `the screen cut off ~${(ou * 100).toFixed(0)}% at the sides, ~${(ov * 100).toFixed(0)}% top and bottom. `
    : 'the screen showed the whole original. ';
  el.textContent = S.edge === 0
    ? cut + `to keep the corners pinned, the outer ${(S.model.margin[0] * 100).toFixed(0)}% / ${(S.model.margin[1] * 100).toFixed(0)}% eases in to the edge.`
    : cut + 'exact leaves that strip transparent (black in video).';
}

function updateOutLine() {
  const el = $('out-line');
  if (!S.orig || !S.film) { el.textContent = 'no source yet'; return; }
  const W = S.orig.w, H = S.orig.h, tl = timeline();
  if (S.film.kind === 'video' && tl) {
    el.textContent = `mp4 \u00b7 ${W}\u00d7${H} \u00b7 ${fmtFps(tl.fps)} fps \u00b7 ${(tl.t1 - tl.t0).toFixed(1)} s`;
  } else {
    el.textContent = `png \u00b7 ${W}\u00d7${H}${S.edge === 1 ? ' \u00b7 alpha edges' : ''}`;
  }
}
const fmtFps = f => (Math.abs(f - Math.round(f)) < 0.01 ? String(Math.round(f)) : f.toFixed(2));

function renderHint() {
  const h = $('stage-hint');
  let msg = '';
  if (!S.orig && !S.film) msg = 'drop the original and the filmed crt shot anywhere';
  else if (!S.orig) msg = 'now the original the crt was showing';
  else if (!S.film) msg = 'now the photo or video of the screen';
  h.hidden = !msg; h.textContent = msg;
}

function refreshAll() {
  updateScore(); renderHand(); updateBandNote(); updateOutLine(); renderHint();
  if (S.model) {
    $('inset').value = String(S.model.inset);
    $('inset-val').textContent = `${(S.model.inset * 100).toFixed(1)}%`;
  }
  pressed('edge-mode', 'edge', S.edge);
  $('undo').disabled = !S.history.length; $('redo').disabled = !S.future.length;
  const ready = !!(S.orig && S.film);
  $('btn-align').disabled = !ready || S.busy;
  for (const id of ['btn-refine', 'btn-export', 'btn-save-json', 'btn-reset-mesh']) {
    $(id).disabled = !S.model || !ready || S.busy;
  }
  for (const b of $('tools').querySelectorAll('button')) b.disabled = b.dataset.tool !== 'view' && !S.model;
  $('led-ready').className = 'led' + (S.model ? ' g' : '');
  $('follow').setAttribute('aria-pressed', String(S.follow));
  paintFader($('inset')); paintFader($('scrub'));
  const tl = timeline();
  $('transport').hidden = !tl;
  $('video-block').hidden = !(S.film && S.film.kind === 'video');
  $('sync-field').hidden = !(S.orig?.kind === 'video' && S.film?.kind === 'video');
  updateOffset();
  if (tl) {
    $('scrub').min = String(tl.t0); $('scrub').max = String(Math.max(tl.t0 + 1e-3, tl.t1)); $('scrub').value = String(S.t);
    $('scrub').step = String(1 / tl.fps);
  }
  updateTimecode();
  updateKeyMark();
}

// ---------------------------------------------------------------- history

function snapshot() {
  return S.model ? { model: modelToJSON(S.model), edge: S.edge, offset: S.offset, score: S.score,
                     hand: S.hand.toJSON(), keyT: S.keyT } : null;
}
function restore(snap) {
  if (!snap) return;
  S.model = modelFromJSON(snap.model); S.edge = snap.edge; S.offset = snap.offset; S.score = snap.score;
  S.hand.fromJSON(snap.hand); S.keyT = snap.keyT ?? S.keyT; S.sel = -1; R.setHand(S.hand.sol);
  R.setModel(S.model); refreshAll(); draw();
}
function pushHistory() { const s = snapshot(); if (!s) return; S.history.push(s); if (S.history.length > 50) S.history.shift(); S.future = []; refreshAll(); }
function undo() { if (!S.history.length || S.busy) return; S.future.push(snapshot()); restore(S.history.pop()); }
function redo() { if (!S.future.length || S.busy) return; S.history.push(snapshot()); restore(S.future.pop()); }

// ---------------------------------------------------------------- loading

async function loadInto(role, file) {
  try {
    busy(true, 'opening', false, file.name);
    const media = isVideoFile(file) ? await loadVideo(file) : await loadImage(file, role);
    S[role] = media;
    S.model = null; S.score = null; S.history = []; S.future = []; S.rasterEdited = false; S.refDetect = null; S.tracked = null;
    S.hand.clear(); S.sel = -1; S.keyT = null; R.setHand(null);
    if (role === 'orig') S.hand.setAspect(media.w / media.h);
    if (role === 'film') S.offset = 0;
    const tl = timeline(); S.t = tl ? (tl.t0 + tl.t1) / 2 : 0;
    R.setModel(null);
    await setSlot(role, media);
    await loadFrames(true);
    if (S.tool !== 'view') setTool('view');
    fitView(); refreshAll(); draw();
    const meta = media.kind === 'video'
      ? `${media.w}×${media.h}, ${fmtFps(media.fps)} fps, ${media.duration.toFixed(1)} s`
      : `${media.w}×${media.h}`;
    const next = !S.orig ? 'now the original.' : !S.film ? 'now the filmed crt shot.'
      : S.orig.kind === 'video' && S.film.kind === 'video' ? 'press auto sync, then align.' : 'press align.';
    lcd(`${media.name}: ${meta}. ${next}`);
  } catch (err) {
    toast(err.message || String(err), true);
  } finally {
    busy(false);
  }
}

async function setSlot(role, media, note = '') {
  const el = $(role === 'orig' ? 'slot-orig' : 'slot-film');
  el.classList.add('loaded');
  el.querySelector('.slot-name').textContent = media.name;
  const meta = media.kind === 'video'
    ? `${media.w}×${media.h} · ${fmtFps(media.fps)} fps · ${media.duration.toFixed(1)} s`
    : `${media.w}×${media.h}`;
  el.querySelector('.slot-meta').textContent = note ? `${meta} · ${note}` : meta;
  const th = el.querySelector('.thumb'), c = th.getContext('2d');
  const src = await media.frame(media.kind === 'video' ? media.duration / 2 : 0);
  c.clearRect(0, 0, th.width, th.height);
  if (src) {
    const k = Math.min(th.width / media.w, th.height / media.h), w = media.w * k, h = media.h * k;
    c.drawImage(src, (th.width - w) / 2, (th.height - h) / 2, w, h);
  }
}

for (const id of ['slot-orig', 'slot-film']) {
  const el = $(id), role = el.dataset.role;
  el.addEventListener('click', () => { if (S.busy) return; pickRole = role; $('file-input').accept = 'image/*,video/*,.mov,.mkv'; $('file-input').click(); });
  el.addEventListener('dragover', e => { e.preventDefault(); el.classList.add('drag'); });
  el.addEventListener('dragleave', () => el.classList.remove('drag'));
  el.addEventListener('drop', e => { e.preventDefault(); el.classList.remove('drag'); const f = e.dataTransfer.files[0]; if (f && !S.busy) loadInto(role, f); });
}
let pickRole = 'orig';
$('file-input').addEventListener('change', e => { const f = e.target.files[0]; e.target.value = ''; if (f) loadInto(pickRole, f); });
stage.addEventListener('dragover', e => { e.preventDefault(); stage.classList.add('drag'); });
stage.addEventListener('dragleave', () => stage.classList.remove('drag'));
stage.addEventListener('drop', async e => {
  e.preventDefault(); stage.classList.remove('drag');
  if (S.busy) return;
  const files = [...e.dataTransfer.files];
  if (files.length >= 2) { await loadInto('orig', files[0]); await loadInto('film', files[1]); return; }
  if (files[0]) await loadInto(!S.orig ? 'orig' : !S.film ? 'film' : 'film', files[0]);
});

// ---------------------------------------------------------------- video timeline

function timeline() {
  const o = S.orig, f = S.film;
  if (o?.kind === 'video') {
    if (f?.kind === 'video') {
      const t0 = Math.max(0, -S.offset), t1 = Math.min(o.duration, f.duration - S.offset);
      return { t0, t1: Math.max(t0, t1), fps: o.fps };
    }
    return { t0: 0, t1: o.duration, fps: o.fps };
  }
  if (f?.kind === 'video') return { t0: 0, t1: f.duration, fps: f.fps };
  return null;
}
const filmTime = t => (S.film?.kind !== 'video' ? 0 : S.orig?.kind === 'video' ? t + S.offset : t);

let frameReq = 0;
async function loadFrames(force = false) {
  const id = ++frameReq;
  try {
    if (S.orig) {
      if (S.orig.kind === 'video' || force) {
        const c = await S.orig.frame(S.t);
        if (id !== frameReq) return;
        if (c) R.setOriginal(c, S.orig.w, S.orig.h);
      }
    }
    if (S.film) {
      if (S.film.kind === 'video' || force) {
        const c = await S.film.frame(filmTime(S.t));
        if (id !== frameReq) return;
        if (c) {
          R.setFilm(c, S.film.w, S.film.h);
          if (S.follow && S.film.kind === 'video' && S.model) S.tracked = trackScreen(c, S.tracked ? 0.5 : 1);
        }
      }
    }
  } catch (err) { toast(err.message || String(err), true); }
  draw();
}

function timecodeOf(t, fps) {
  const f = Math.round(t * fps), r = Math.round(fps);
  const fr = f % r, sec = Math.floor(f / r), p = n => String(n).padStart(2, '0');
  return `${p(Math.floor(sec / 3600))}:${p(Math.floor(sec / 60) % 60)}:${p(sec % 60)}:${p(fr)}`;
}
function updateTimecode() {
  const tl = timeline();
  if (!tl) return;
  const fps = tl.fps, f = Math.round(S.t * fps), r = Math.round(fps);
  const fr = f % r, sec = Math.floor(f / r), p = n => String(n).padStart(2, '0');
  $('timecode').textContent = `${p(Math.floor(sec / 3600))}:${p(Math.floor(sec / 60) % 60)}:${p(sec % 60)}:${p(fr)}`;
}
$('scrub').addEventListener('input', e => { S.t = Number(e.target.value); updateTimecode(); loadFrames(); });
function stepFrame(d) {
  const tl = timeline(); if (!tl) return;
  S.t = Math.max(tl.t0, Math.min(tl.t1, S.t + d / tl.fps));
  $('scrub').value = String(S.t); updateTimecode(); loadFrames();
}
$('play').addEventListener('click', () => {
  S.playing = !S.playing;
  $('play-icon').setAttribute('d', S.playing ? 'M7 5h4v14H7zM13 5h4v14h-4z' : 'M8 5v14l11-7z');
  if (!S.playing) return;
  let last = performance.now();
  const tick = async now => {
    if (!S.playing) return;
    const tl = timeline();
    S.t += (now - last) / 1000; last = now;
    if (S.t > tl.t1) S.t = tl.t0;
    $('scrub').value = String(S.t); updateTimecode();
    await loadFrames();
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
});

function updateOffset() {
  const fps = S.orig?.fps || 30, fr = Math.round(S.offset * fps);
  $('offset-val').textContent = `${S.offset >= 0 ? '+' : '−'}${Math.abs(S.offset).toFixed(3)} s · ${fr >= 0 ? '+' : '−'}${Math.abs(fr)} fr`;
}
function nudgeOffset(d) {
  const fps = S.film?.fps || 30;
  S.offset += d / fps; updateOffset(); refreshAll(); loadFrames();
}

async function autoSync() {
  if (!(S.orig?.kind === 'video' && S.film?.kind === 'video')) return;
  S.cancel = false;
  try {
    busy(true, 'syncing', true, 'reading the original\u2019s brightness, frame by frame');
    const a = await lumaSignature(S.orig, f => progress(0.02 + 0.28 * f));
    if (S.cancel) throw new Error('cancelled');
    log('reading the filmed clip\u2019s brightness');
    const b = await lumaSignature(S.film, f => progress(0.3 + 0.28 * f));
    const cands = syncCandidates(a, S.orig.fps, b, S.film.fps, 4).slice(0, 3);
    if (!cands.length) throw new Error('The clips are too short to sync.');
    // brightness alone can lock onto a repeat: check each candidate on a real frame
    let best = null;
    for (let i = 0; i < cands.length; i++) {
      const c = cands[i];
      const t0 = Math.max(0, -c.offset), t1 = Math.min(S.orig.duration, S.film.duration - c.offset);
      if (t1 - t0 < 0.5) continue;
      log(`checking candidate ${i + 1} of ${cands.length} (${c.offset >= 0 ? '+' : ''}${c.offset.toFixed(2)} s)`);
      progress(0.6 + 0.08 * i);
      const tm = (t0 + t1) / 2;
      const o = await lumaAt(S.orig, tm, 'orig'), f = await lumaAt(S.film, tm + c.offset, 'film');
      const r = await engine.runAlign({ orig: o, film: f, W: S.orig.w, H: S.orig.h, FW: S.film.w, FH: S.film.h, quality: 'quick' });
      if (!best || r.score > best.score) best = { ...c, score: r.score, tm, model: r.model, orig: o };
    }
    if (!best) throw new Error('No usable overlap between the clips.');
    // frame-accurate: hold that alignment and score the neighbouring frame offsets
    const step = 1 / S.film.fps;
    let fine = { offset: best.offset, score: -1 };
    for (let k = -6; k <= 6; k++) {
      if (S.cancel) throw new Error('cancelled');
      const off = best.offset + k * step;
      const tf = best.tm + off;
      if (tf < 0 || tf > S.film.duration) continue;
      log(`fine sync \u00b7 ${off >= 0 ? '+' : ''}${off.toFixed(3)} s`);
      progress(0.85 + 0.15 * (k + 6) / 12);
      const f = await lumaAt(S.film, tf, 'film');
      const r = await engine.runScore({ model: best.model, orig: best.orig, film: f, w: 384 });
      if (r.score > fine.score) fine = { offset: off, score: r.score };
    }
    S.offset = fine.offset;
    const tl = timeline(); S.t = (tl.t0 + tl.t1) / 2;
    lcd(`synced: the filmed clip ${S.offset >= 0 ? 'lags' : 'leads'} by ${Math.abs(S.offset).toFixed(3)} s. press align.`);
    refreshAll(); await loadFrames();
  } catch (err) { if (err.message !== 'cancelled') toast(err.message || String(err), true); else toast('Sync cancelled.'); }
  finally { busy(false); }
}

// Follow camera: re-detect the screen on each frame and move the pinned corners by
// how much the detection moved since the frame the alignment was made on.
function frameLuma(src, w, h, maxW = 480) {
  const k = Math.min(1, maxW / w), cw = Math.round(w * k), ch = Math.round(h * k);
  const c = document.createElement('canvas'); c.width = cw; c.height = ch;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(src, 0, 0, cw, ch);
  return lumaFromRGBA(ctx.getImageData(0, 0, cw, ch).data, cw, ch);
}
function detectFull(src) {
  const lum = frameLuma(src, S.film.w, S.film.h);
  const d = detectScreen(lum);
  const k = S.film.w / lum.w;
  return d.ok ? d.corners.map(p => [p[0] * k, p[1] * k]) : null;
}
function trackScreen(src, alpha) {
  if (!S.refDetect) return S.tracked;
  const det = detectFull(src);
  if (!det) return S.tracked;
  const base = S.model.screen.corners;
  const target = base.map((c, i) => [c[0] + det[i][0] - S.refDetect[i][0], c[1] + det[i][1] - S.refDetect[i][1]]);
  const prev = S.tracked ? S.tracked.corners : target;
  const next = target.map((c, i) => [prev[i][0] + alpha * (c[0] - prev[i][0]), prev[i][1] + alpha * (c[1] - prev[i][1])]);
  return makeScreen(next, S.model.screen.bulge);
}

// ---------------------------------------------------------------- alignment

const STAGES = { screen: 'finding the screen', search: 'searching zoom and offset', tournament: 'comparing candidates' };
function onProgress(e) {
  let text = STAGES[e.stage] || e.stage, f = 0.05;
  if (e.stage === 'search') f = 0.05 + 0.2 * ((e.iter || 0) / (e.of || 2));
  const cand = /^candidate (\d+)\/(\d+), level (\d+)\/(\d+)/.exec(e.stage);
  if (cand) { const [, c, n, l] = cand.map(Number); f = 0.25 + 0.4 * ((c - 1 + (l - 1) / 2 + (e.iter || 0) / (e.of || 1) / 2) / n); text = `testing candidate ${c} of ${n}`; }
  const lev = /^(level|refine) (\d+)\/(\d+)/.exec(e.stage);
  if (lev) { const l = Number(lev[2]), n = Number(lev[3]); f = (lev[1] === 'refine' ? 0.1 : 0.65) + (lev[1] === 'refine' ? 0.9 : 0.35) * ((l - 1 + (e.iter || 0) / (e.of || 1)) / n); text = `fitting the mesh \u00b7 pass ${l} of ${n}`; }
  if (e.stage === 'tournament') return;
  log(text); progress(f);
}

async function runAlign(kind) {
  if (!S.orig || !S.film || S.busy) return;
  if (kind === 'refine' && !S.model) kind = 'align';
  S.playing = false;
  let pushed = false;
  try {
    S.cancel = false;
    busy(true, kind === 'align' ? 'aligning' : 'refining', true, 'reading frames');
    const origLum = S.orig.kind === 'image' ? S.orig.lum : await lumaAt(S.orig, S.t, 'orig');
    const filmLum = S.film.kind === 'image' ? S.film.lum : await lumaAt(S.film, filmTime(S.t), 'film');
    if (kind === 'align') {
      if (S.model) { pushHistory(); pushed = true; }
      const screen = S.rasterEdited && S.model
        ? { corners: S.model.screen.corners, bulge: S.model.screen.bulge, inset: S.model.inset } : null;
      const pins = S.model && S.rasterEdited ? S.model.pins : [];
      const res = await engine.runAlign({ orig: origLum, film: filmLum, W: S.orig.w, H: S.orig.h, FW: S.film.w, FH: S.film.h,
                                          quality: S.quality, lambda: S.lambda, screen }, onProgress);
      S.model = res.model; S.score = res.score;
      S.hand.clear(); S.sel = -1;
      if (pins.length) {
        S.model.pins = pins;
        const r2 = await engine.runRefine({ model: S.model, orig: origLum, film: filmLum, quality: S.quality, lambda: S.lambda }, onProgress);
        S.model = r2.model; S.score = r2.score;
      }
      if (!res.screenDetected) toast('Couldn’t find the screen outline, so a default frame was used. Check it with Edit screen corners.', true);
    } else {
      const res = await engine.runRefine({ model: S.model, orig: origLum, film: filmLum, quality: S.quality, lambda: S.lambda,
                                           fromLevel: S.rasterEdited ? 0 : null }, onProgress);
      S.model = res.model; S.score = res.score;
    }
    S.rasterEdited = false; $('btn-refine').classList.remove('nudge');
    R.setModel(S.model); R.setHand(S.hand.sol);
    markKey();
    statusLine(kind === 'align' ? 'done' : 'refined');
    if (S.film.kind === 'video') {
      const src = await S.film.frame(filmTime(S.t));
      S.refDetect = src ? detectFull(src) : null; S.tracked = null;
    }
    refreshAll(); draw();
  } catch (err) {
    if (err.message === 'cancelled') { if (pushed) S.history.pop(); toast('Stopped. The previous alignment is kept.'); }
    else { console.error(err); toast(err.message || String(err), true); }
  } finally {
    busy(false);
  }
}

// ---------------------------------------------------------------- export

const baseName = n => n.replace(/\.[^.]+$/, '');

async function exportLayer() {
  if (!S.model || S.busy) return;
  const asVideo = S.film.kind === 'video' && !!timeline();
  S.cancel = false;
  try {
    busy(true, 'exporting', asVideo, asVideo ? 'preparing the encoder' : 'rendering the layer');
    const r = exportRenderer();
    r.setModel(S.model);
    r.setHand(S.hand.sol);
    const W = S.orig.w, H = S.orig.h;
    if (W > r.maxTex || H > r.maxTex) throw new Error(`This GPU renders at most ${r.maxTex} px per side; the original is ${W}×${H}.`);
    let blob, name;
    if (!asVideo) {
      const src = S.film.kind === 'image' ? S.film.bitmap : await S.film.frame(filmTime(S.t));
      r.setFilm(src, S.film.w, S.film.h);
      r.renderLayer(W, H, { edge: S.edge, screenOverride: S.follow ? S.tracked : null });
      blob = await new Promise((res, rej) => r.canvas.toBlob(b => (b ? res(b) : rej(new Error('PNG encoding failed.'))), 'image/png'));
      name = `${baseName(S.film.name)}_glitch-layer.png`;
    } else {
      blob = await renderVideo(r, W, H);
      name = `${baseName(S.film.name)}_glitch-layer.mp4`;
    }
    log('saving');
    const st = await saveFile(name, blob);
    toast(st === 'declined' ? 'Save cancelled.' : `Exported ${name} (${(blob.size / 1048576).toFixed(1)} MB).`);
    if (st !== 'declined') lcd(`saved ${name}, ${(blob.size / 1048576).toFixed(1)} MB.`);
  } catch (err) {
    if (!S.cancel) { console.error(err); toast(err.message || String(err), true); } else toast('Export cancelled.');
  } finally {
    busy(false);
  }
}

async function renderVideo(r, W, H) {
  const M = await mb();
  const tl = timeline(), fps = tl.fps;
  const n = Math.max(1, Math.floor((tl.t1 - tl.t0) * fps + 1e-6));
  const codec = await M.getFirstEncodableVideoCodec(['avc', 'vp9', 'av1', 'hevc'], { width: W, height: H, frameRate: fps });
  if (!codec) throw new Error(`This browser can't encode ${W}×${H} video. Try Chrome or Edge on a desktop.`);
  const output = new M.Output({ format: new M.Mp4OutputFormat({ fastStart: 'in-memory' }), target: new M.BufferTarget() });
  const bitrate = Math.round(Math.min(80e6, Math.max(8e6, W * H * fps * 0.12)));
  const src = new M.CanvasSource(r.canvas, { codec, bitrate });
  output.addVideoTrack(src, { frameRate: fps });
  await output.start();
  const film = S.film, origin = film.start;
  const it = film.sink.canvases(origin + Math.max(0, filmTime(tl.t0)))[Symbol.asyncIterator]();
  let cur = (await it.next()).value, nxt = (await it.next()).value, shown = null;
  S.tracked = null;
  let screen = null;
  try {
    for (let i = 0; i < n; i++) {
      if (S.cancel) { await output.cancel(); throw new Error('cancelled'); }
      const target = origin + filmTime(tl.t0 + i / fps);
      while (nxt && nxt.timestamp <= target + 1e-4) { cur = nxt; nxt = (await it.next()).value; }
      if (!cur) break;
      if (shown !== cur) {
        r.setFilm(cur.canvas, film.w, film.h);
        if (S.follow) { S.tracked = trackScreen(cur.canvas, S.tracked ? 0.35 : 1); screen = S.tracked; }
        shown = cur;
      }
      r.renderLayer(W, H, { edge: S.edge, screenOverride: screen });
      await src.add(i / fps, 1 / fps);
      if (i % 4 === 0) { $('busy-stage').textContent = `Encoding ${codec.toUpperCase()} · frame ${i + 1} of ${n}`; progress((i + 1) / n); }
    }
  } finally {
    await it.return?.(); // release the decoder and any frame it decoded ahead
  }
  $('busy-stage').textContent = 'Finishing the file';
  await output.finalize();
  return new Blob([output.target.buffer], { type: 'video/mp4' });
}

// ---------------------------------------------------------------- alignment files

async function saveAlignment() {
  if (!S.model) return;
  const data = {
    app: 'crt-align', version: 1,
    original: { name: S.orig.name, w: S.orig.w, h: S.orig.h },
    filmed: { name: S.film.name, w: S.film.w, h: S.film.h },
    edge: S.edge, offset: S.offset, quality: S.quality, lambda: S.lambda, score: S.score,
    hand: S.hand.toJSON(), keyT: S.keyT,
    model: modelToJSON(S.model),
  };
  const name = `${baseName(S.film.name)}_alignment.json`;
  try {
    const st = await saveFile(name, new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' }));
    if (st !== 'declined') toast(`Saved ${name}. Load it on another shot from the same setup, then Refine.`);
  } catch (err) { toast(err.message, true); }
}

async function loadAlignment(file) {
  try {
    const j = JSON.parse(await file.text());
    if (!['crt-align', 'crt-match'].includes(j.app) || !j.model) throw new Error('That file isn’t a CRT Align alignment.');
    if (!S.orig || !S.film) throw new Error('Load the original and the filmed capture first, then the alignment.');
    if (S.model) pushHistory();
    const mj = { ...j.model };
    // same camera setup at another resolution: scale the screen geometry
    const kx = S.film.w / mj.FW, ky = S.film.h / mj.FH;
    mj.corners = mj.corners.map(p => [p[0] * kx, p[1] * ky]);
    mj.bulge = mj.bulge.map(b => b * (kx + ky) / 2);
    mj.FW = S.film.w; mj.FH = S.film.h; mj.W = S.orig.w; mj.H = S.orig.h;
    S.model = modelFromJSON(mj);
    S.edge = j.edge ?? 0; S.score = null;
    S.hand.setAspect(S.orig.w / S.orig.h); S.hand.fromJSON(j.hand); S.sel = -1; R.setHand(S.hand.sol);
    if (typeof j.keyT === 'number') S.keyT = j.keyT;
    if (typeof j.offset === 'number' && S.film.kind === 'video' && S.orig.kind === 'video') S.offset = j.offset;
    R.setModel(S.model); refreshAll(); draw();
    toast('Alignment loaded. Press Refine to fit it to this shot.');
  } catch (err) { toast(err.message || String(err), true); }
}

// ---------------------------------------------------------------- busy / toast

function busy(on, label = '', cancellable = false, detail = '') {
  S.busy = on;
  $('busy').hidden = !on;
  $('led-busy').className = 'led' + (on ? ' p pulse' : '');
  if (on) {
    $('busy-label').textContent = label;
    $('busy-stage').textContent = detail;
    progress(0.02);
    $('busy-cancel').hidden = !cancellable;
  }
  refreshAll();
}
const log = t => { $('busy-stage').textContent = t; };
function progress(f) { $('busy-bar').style.width = `${Math.round(Math.max(0, Math.min(1, f)) * 100)}%`; }

// the LCD carries the current status, like Code Painter's readout
function lcd(msg, err = false) { const el = $('lcd'); el.textContent = msg; el.classList.toggle('err', err); }

function toast(msg, error = false) {
  const box = $('toasts'), t = document.createElement('div');
  t.className = 'toast' + (error ? ' err' : '');
  t.textContent = msg;
  box.append(t);
  while (box.children.length > 3) box.firstChild.remove();
  setTimeout(() => t.remove(), error ? 9000 : 5000);
  if (error) lcd(msg, true);
}
function statusLine(prefix = 'done') {
  if (!S.model || S.score == null) return;
  const pins = S.hand.count;
  lcd(`${prefix}: edge match ${Math.round(S.score * 100)}%, mesh ${S.model.nx}\u00d7${S.model.ny}, ${pins} pin${pins === 1 ? '' : 's'}${S.edge ? ', exact edges' : ''}.`);
}
window.addEventListener('unhandledrejection', e => { console.error(e.reason); toast(e.reason?.message || String(e.reason), true); });

// ---------------------------------------------------------------- start

async function loadSample(force = false) {
  if (force) {
    S.orig = S.film = S.model = null; S.score = null; S.history = []; S.future = []; S.offset = 0;
    S.rasterEdited = false; S.follow = false; R.setModel(null);
    S.hand.clear(); S.sel = -1; S.keyT = null; R.setHand(null);
    if (S.tool !== 'view') setTool('view');
  }
  try {
    const [o, f, a] = await Promise.all([
      fetch('sample/original.jpg').then(r => (r.ok ? r.blob() : null)),
      fetch('sample/filmed.jpg').then(r => (r.ok ? r.blob() : null)),
      fetch('sample/alignment.json').then(r => (r.ok ? r.json() : null)),
    ]);
    if (!o || !f || S.orig || S.film) return;
    S.orig = await loadImage(new File([o], 'sample-original.jpg', { type: 'image/jpeg' }), 'orig');
    S.film = await loadImage(new File([f], 'sample-filmed-crt.jpg', { type: 'image/jpeg' }), 'film');
    R.setOriginal(S.orig.bitmap, S.orig.w, S.orig.h);
    S.hand.setAspect(S.orig.w / S.orig.h);
    R.setFilm(S.film.bitmap, S.film.w, S.film.h);
    await setSlot('orig', S.orig, 'sample'); await setSlot('film', S.film, 'sample');
    if (a && a.model) {
      S.model = modelFromJSON(a.model); S.score = a.score ?? null; S.edge = a.edge ?? 0;
      R.setModel(S.model);
    }
    fitView(); refreshAll(); draw();
    lcd(`sample loaded: cartoon on a crt, edge match ${Math.round((S.score ?? 0) * 100)}%. drop your own original and filmed shot to start.`);
  } catch (err) { console.warn('sample not loaded', err); refreshAll(); }
}

setMode(1);
setTool('view');
pressed('quality', 'q', S.quality);
S.lambda = Math.pow(10, Number($('stiffness').value));
refreshAll();
prepareDownloads();
loadSample();
