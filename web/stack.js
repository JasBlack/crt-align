// Stack mode UI: load a set of photos, line them up on one anchor, find the crop
// every photo can fill, export the crops.
import { StackGL } from './stackgl.js';
import { saveFile, prepareDownloads, mb } from './media.js';
import { lumaFromRGBA } from './core/image.js';
import { FINE, brightestSpot, cropReport, leaveOneOut, footprint, pinAnchor, reduceToTranslation, nativeScale, ASPECTS } from './core/stack.js';
import { I3, mul3, inv3, apply, translation, scaling, sane, inside } from './core/geom.js';
import { ZipWriter } from './zip.js';

const $ = id => document.getElementById(id);
const PREVIEW = 1600;   // display texture long side

// ---------------------------------------------------------------- theme
let C = {};
function readTheme() {
  const cs = getComputedStyle(document.body), g = n => cs.getPropertyValue(n).trim();
  C = { accent: g('--accent'), accent2: g('--accent2'), red: g('--red'), muted: g('--muted'), light: document.body.dataset.ui === 'light' };
}
function setTheme(ui) {
  document.body.dataset.ui = ui;
  try { localStorage.setItem('align-ui', ui); } catch { /* storage blocked */ }
  readTheme(); draw();
}
try { document.body.dataset.ui = localStorage.getItem('align-ui') || localStorage.getItem('crt-align-ui') || 'dark'; } catch { /* storage blocked */ }
readTheme();

// settings remembered between visits (versioned key, family rule)
const PREFS = 'align-stack-v1';
const PREF_KEYS = ['method', 'kind', 'aspect', 'place', 'size', 'fmt', 'fps', 'hold'];
function loadPrefs() {
  try { return JSON.parse(localStorage.getItem(PREFS) || '{}') || {}; } catch { return {}; }
}
function savePrefs() {
  try { localStorage.setItem(PREFS, JSON.stringify(Object.fromEntries(PREF_KEYS.map(k => [k, S[k]])))); } catch { /* storage blocked */ }
}

let nextId = 1;
const S = {
  items: [], ref: 0, cur: 0,
  anchor: null, roi: null,
  method: 'match', kind: 'homography', tool: 'anchor', view: 'aligned',
  aspect: '16:9', place: 'centre', zoom: 1, size: 'native', fmt: 'jpg',
  vp: null, rawVp: null, aligned: false, busy: false, cancel: false,
  report: null, costs: [], playing: false,
  fps: 24, hold: 2,
  manual: null,          // crop placed by hand: { aspect, rect } in reference px
};
{
  const p = loadPrefs(), ok = {
    method: ['match', 'bright', 'manual'], kind: ['translate', 'similarity', 'homography'],
    aspect: ASPECTS.map(a => a.id), place: ['centre', 'thirds', 'free'], size: ['native', '1080', '2160'], fmt: ['jpg', 'png', 'mp4'], fps: [12, 24, 25, 30],
    hold: [...Array(30).keys()].map(k => k + 1),
  };
  for (const k of PREF_KEYS) if (ok[k].includes(p[k])) S[k] = p[k];
  if (S.method === 'manual') S.tool = 'anchor';
}

// ---------------------------------------------------------------- canvases

const stage = $('stage'), glCanvas = $('gl'), ov = $('overlay');
let G;
try { G = new StackGL(glCanvas); } catch (e) { $('stage-hint').textContent = e.message; throw e; }
const octx = ov.getContext('2d');
let dpr = 1, VW = 1, VH = 1;

function resize() {
  const r = stage.getBoundingClientRect();
  dpr = window.devicePixelRatio || 1;
  VW = glCanvas.width = ov.width = Math.max(1, Math.round(r.width * dpr));
  VH = glCanvas.height = ov.height = Math.max(1, Math.round(r.height * dpr));
  if (!S.vp) fitView();
  draw();
}
new ResizeObserver(resize).observe(stage);

const R = () => S.items[S.ref];
const CUR = () => S.items[S.cur];
const isRaw = () => S.view === 'raw' || (S.method === 'manual' && S.tool === 'anchor' && S.cur !== S.ref);
const curVp = () => (isRaw() ? S.rawVp : S.vp);

// the map actually used for item i: fitted (or a plain scale before aligning) + nudge
function effMap(i) {
  const it = S.items[i], ref = R();
  if (!it || !ref) return I3();
  let m = it.map || scaling(it.W / ref.W);
  if (it.nudge) m = mul3(m, translation(-it.nudge[0], -it.nudge[1]));
  return m;
}
const usable = i => S.items[i] && S.items[i].include && (i === S.ref || (S.items[i].map && S.items[i].res?.ok));
const anchorPt = () => S.anchor || (R() ? [R().W / 2, R().H / 2] : [0, 0]);

function fitTo(x0, y0, x1, y1, pad = 0.94) {
  const w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0);
  const z = Math.min(VW / w, VH / h) * pad;
  return { x: (VW - w * z) / 2 - x0 * z, y: (VH - h * z) / 2 - y0 * z, z };
}
function fitView() {
  const ref = R();
  if (!ref) { S.vp = { x: 0, y: 0, z: 1 }; S.rawVp = S.vp; updateZoom(); return; }
  if (S.view === 'crop' && S.report) {
    const row = S.report.rows.find(r => r.id === S.aspect);
    if (row?.rect) S.vp = fitTo(row.rect.x, row.rect.y, row.rect.x + row.rect.w, row.rect.y + row.rect.h, 0.98);
  } else {
    // everything lined up, but not more than 1.6x the reference frame
    let x0 = 0, y0 = 0, x1 = ref.W, y1 = ref.H;
    if (S.aligned) {
      S.items.forEach((it, i) => {
        if (!usable(i)) return;
        const q = footprint(effMap(i), it.W, it.H);
        if (!q) return;
        for (const [x, y] of q) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
      });
      x0 = Math.max(x0, -0.3 * ref.W); y0 = Math.max(y0, -0.3 * ref.H);
      x1 = Math.min(x1, 1.3 * ref.W); y1 = Math.min(y1, 1.3 * ref.H);
    }
    S.vp = fitTo(x0, y0, x1, y1);
  }
  const c = CUR();
  if (c) S.rawVp = fitTo(0, 0, c.W, c.H);
  updateZoom();
}
function updateZoom() {
  const v = curVp(), it = isRaw() ? CUR() : R();
  if (!v || !it) { $('zoom-readout').textContent = '100%'; return; }
  // zoom relative to the photo's real pixels
  const s = isRaw() ? 1 : 1;
  $('zoom-readout').textContent = `${Math.round((v.z / dpr) * s * 100)}%`;
}

const evPos = e => { const r = stage.getBoundingClientRect(); return [(e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr]; };
const toSpace = ([bx, by], v = curVp()) => [(bx - v.x) / v.z, (by - v.y) / v.z];
const toBuf = (x, y, v = curVp()) => [x * v.z + v.x, y * v.z + v.y];

// ---------------------------------------------------------------- drawing

let raf = 0;
function draw() { if (!raf) raf = requestAnimationFrame(() => { raf = 0; paint(); }); }

const layer = i => {
  const it = S.items[i];
  return it?.tex ? { tex: it.tex, map: effMap(i), w: it.W, h: it.H } : null;
};

function paint() {
  G.begin(VW, VH);
  octx.clearRect(0, 0, VW, VH);
  const ref = R(), cur = CUR();
  $('stage-hint').hidden = !!ref;
  if (!ref || !cur) return;
  if (isRaw()) {
    if (cur.tex) G.draw({ tex: cur.tex, map: I3(), w: cur.W, h: cur.H }, S.rawVp);
    return paintRawOverlay();
  }
  const v = S.vp;
  if (S.view === 'mean') {
    const ls = S.items.map((_, i) => (usable(i) ? layer(i) : null)).filter(Boolean);
    G.mean(ls, v);
  } else if (S.view === 'diff' && S.cur !== S.ref) {
    const a = layer(S.cur), b = layer(S.ref);
    if (a && b) G.diff(a, b, v);
  } else if (S.view === 'onion' && S.cur !== S.ref) {
    const b = layer(S.ref), a = layer(S.cur);
    if (b) G.draw(b, v);
    if (a) G.draw(a, v, { alpha: 0.5 });
  } else {
    const a = layer(S.cur);
    if (a) G.draw(a, v);
  }
  paintOverlay();
}

function poly(P, v = S.vp) {
  octx.beginPath();
  P.forEach(([x, y], k) => { const [bx, by] = toBuf(x, y, v); if (k) octx.lineTo(bx, by); else octx.moveTo(bx, by); });
  octx.closePath();
}

function cross(x, y, col, r = 9) {
  const R_ = r * dpr;
  octx.lineWidth = 3.5 * dpr; octx.strokeStyle = 'rgba(0,0,0,0.6)';
  const path = () => {
    octx.beginPath(); octx.arc(x, y, R_, 0, Math.PI * 2);
    octx.moveTo(x - R_ * 1.8, y); octx.lineTo(x - R_ * 0.45, y); octx.moveTo(x + R_ * 0.45, y); octx.lineTo(x + R_ * 1.8, y);
    octx.moveTo(x, y - R_ * 1.8); octx.lineTo(x, y - R_ * 0.45); octx.moveTo(x, y + R_ * 0.45); octx.lineTo(x, y + R_ * 1.8);
  };
  path(); octx.stroke();
  octx.lineWidth = 1.6 * dpr; octx.strokeStyle = col; path(); octx.stroke();
}

function paintOverlay() {
  const v = S.vp, ref = R();
  const row = S.aligned && S.report ? S.report.rows.find(r => r.id === S.aspect) : null;
  // dim everything outside the crop
  if (row?.rect) {
    const r = row.rect, [x0, y0] = toBuf(r.x, r.y, v), [x1, y1] = toBuf(r.x + r.w, r.y + r.h, v);
    octx.fillStyle = S.view === 'crop' ? 'rgba(11,11,12,0.96)' : 'rgba(0,0,0,0.5)';  // the stage is dark in both themes
    octx.beginPath(); octx.rect(0, 0, VW, VH); octx.rect(x0, y0, x1 - x0, y1 - y0); octx.fill('evenodd');
  }
  if (S.view !== 'crop') {
    // reference frame
    octx.setLineDash([4 * dpr, 4 * dpr]); octx.lineWidth = 1 * dpr; octx.strokeStyle = C.muted;
    poly([[0, 0], [ref.W, 0], [ref.W, ref.H], [0, ref.H]]); octx.stroke();
    // the area every photo covers
    if (S.aligned && S.report?.region?.length) {
      octx.strokeStyle = C.accent; octx.globalAlpha = 0.8;
      poly(S.report.region); octx.stroke(); octx.globalAlpha = 1;
    }
    octx.setLineDash([]);
    // current photo's outline
    if (S.cur !== S.ref) {
      const q = footprint(effMap(S.cur), CUR().W, CUR().H);
      if (q) { octx.strokeStyle = C.accent2; octx.lineWidth = 1.5 * dpr; poly(q); octx.stroke(); }
    }
  }
  if (row?.rect) {
    const r = row.rect, [x0, y0] = toBuf(r.x, r.y, v), [x1, y1] = toBuf(r.x + r.w, r.y + r.h, v);
    octx.strokeStyle = '#f4f5f7'; octx.lineWidth = 1.5 * dpr; octx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    // thirds
    octx.globalAlpha = 0.28; octx.lineWidth = 1 * dpr; octx.beginPath();
    for (const f of [1 / 3, 2 / 3]) {
      octx.moveTo(x0 + (x1 - x0) * f, y0); octx.lineTo(x0 + (x1 - x0) * f, y1);
      octx.moveTo(x0, y0 + (y1 - y0) * f); octx.lineTo(x1, y0 + (y1 - y0) * f);
    }
    octx.stroke(); octx.globalAlpha = 1;
    if (S.tool === 'frame') {
      octx.fillStyle = '#f4f5f7'; octx.strokeStyle = 'rgba(0,0,0,0.6)'; octx.lineWidth = 1 * dpr;
      const hs = 4.5 * dpr;
      for (const [x, y] of [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) {
        octx.fillRect(x - hs, y - hs, hs * 2, hs * 2); octx.strokeRect(x - hs, y - hs, hs * 2, hs * 2);
      }
    }
    if (S.view !== 'crop') {
      const label = `${row.id}  ${outSize(row).join('\u00d7')}${row.manual ? '  by hand' : ''}`;
      octx.font = `600 ${10 * dpr}px "IBM Plex Mono", monospace`;
      const tw = octx.measureText(label).width + 12 * dpr;
      octx.fillStyle = 'rgba(11,11,12,0.85)'; octx.fillRect(x0, y0 - 18 * dpr, tw, 18 * dpr);
      octx.fillStyle = '#f4f5f7'; octx.fillText(label, x0 + 6 * dpr, y0 - 5 * dpr);
    }
  }
  // subject box
  if (S.roi && S.method === 'match' && S.view !== 'crop') {
    const r = S.roi, [x0, y0] = toBuf(r.x, r.y, v), [x1, y1] = toBuf(r.x + r.w, r.y + r.h, v);
    octx.setLineDash([6 * dpr, 4 * dpr]); octx.strokeStyle = C.accent2; octx.lineWidth = 1.5 * dpr;
    octx.strokeRect(x0, y0, x1 - x0, y1 - y0); octx.setLineDash([]);
  }
  if (dragBox) {
    const [x0, y0] = toBuf(dragBox.x, dragBox.y, v), [x1, y1] = toBuf(dragBox.x + dragBox.w, dragBox.y + dragBox.h, v);
    octx.setLineDash([6 * dpr, 4 * dpr]); octx.strokeStyle = C.accent2; octx.lineWidth = 1.5 * dpr;
    octx.strokeRect(x0, y0, x1 - x0, y1 - y0); octx.setLineDash([]);
  }
  const [ax, ay] = anchorPt();
  if (S.anchor || S.aligned) cross(...toBuf(ax, ay, v), C.accent2);
}

function paintRawOverlay() {
  const v = S.rawVp, it = CUR();
  octx.setLineDash([4 * dpr, 4 * dpr]); octx.lineWidth = 1 * dpr; octx.strokeStyle = C.muted;
  poly([[0, 0], [it.W, 0], [it.W, it.H], [0, it.H]], v); octx.stroke(); octx.setLineDash([]);
  if (S.method === 'manual') {
    if (S.cur === S.ref) { cross(...toBuf(...anchorPt(), v), C.accent2); return; }
    if (it.manual) cross(...toBuf(...it.manual, v), C.accent2);
    else if (S.aligned || S.anchor) {
      // where the current fit puts the anchor: a faint guess to click near
      const [u, w] = apply(effMap(S.cur), ...anchorPt());
      octx.globalAlpha = 0.45; cross(...toBuf(u, w, v), C.muted); octx.globalAlpha = 1;
    }
  } else if (S.anchor || S.aligned) {
    const [u, w] = apply(effMap(S.cur), ...anchorPt());
    cross(...toBuf(u, w, v), C.accent2);
  }
}

// ---------------------------------------------------------------- interaction

let drag = null, dragBox = null, spaceDown = false;

stage.addEventListener('wheel', e => {
  e.preventDefault();
  const v = curVp(); if (!v) return;
  const [bx, by] = evPos(e);
  const k = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015));
  const z = Math.max(0.005, Math.min(40, v.z * k)), f = z / v.z;
  v.x = bx - (bx - v.x) * f; v.y = by - (by - v.y) * f; v.z = z;
  updateZoom(); draw();
}, { passive: false });
stage.addEventListener('contextmenu', e => e.preventDefault());

stage.addEventListener('pointerdown', e => {
  if (S.busy || !R()) return;
  stage.setPointerCapture(e.pointerId);
  const pos = evPos(e), p = toSpace(pos);
  const pan = e.button === 1 || e.button === 2 || spaceDown || S.tool === 'view';
  if (pan) { drag = { kind: 'pan', pos, v: { ...curVp() } }; stage.classList.add('panning'); return; }
  if (S.tool === 'nudge' && !isRaw() && S.cur !== S.ref) {
    const it = CUR();
    drag = { kind: 'nudge', p, start: it.nudge ? [...it.nudge] : [0, 0] };
    stage.classList.add('panning');
    return;
  }
  if (S.tool === 'frame' && !isRaw()) {
    const row = curRow();
    if (!row?.rect) { toast('Align first: the crop appears once the photos are lined up.'); return; }
    const r = row.rect, v = S.vp, hit = 12 * dpr;
    const corners = [[r.x, r.y], [r.x + r.w, r.y], [r.x + r.w, r.y + r.h], [r.x, r.y + r.h]];
    const ci = corners.findIndex(([x, y]) => { const [bx, by] = toBuf(x, y, v); return Math.hypot(bx - pos[0], by - pos[1]) < hit; });
    if (ci >= 0) drag = { kind: 'crop-size', fixed: corners[(ci + 2) % 4], sx: ci === 1 || ci === 2 ? 1 : -1, sy: ci >= 2 ? 1 : -1, start: { ...r } };
    else if (p[0] >= r.x && p[0] <= r.x + r.w && p[1] >= r.y && p[1] <= r.y + r.h) { drag = { kind: 'crop-move', p, start: { ...r } }; stage.classList.add('panning'); }
    return;
  }
  if (S.tool === 'anchor') drag = { kind: 'anchor', pos, p, raw: isRaw() };
});

const curRow = () => (S.aligned && S.report ? S.report.rows.find(r => r.id === S.aspect) : null);

// slide as far as the common area allows (each axis on its own, so it runs along edges)
function moveCrop(start, dx, dy) {
  const V = S.report.region;
  let r = { ...start };
  for (const [ax, d] of [['x', dx], ['y', dy]]) {
    const at = t => ({ ...r, [ax]: r[ax] + d * t });
    if (rectFits(V, at(1))) { r = at(1); continue; }
    let lo = 0, hi = 1;
    for (let k = 0; k < 24; k++) { const m = (lo + hi) / 2; if (rectFits(V, at(m))) lo = m; else hi = m; }
    r = at(lo);
  }
  return r;
}
// grow / shrink from the opposite corner, shape locked
function sizeCrop(d, p) {
  const V = S.report.region, ar = d.start.w / d.start.h, [fx, fy] = d.fixed;
  const want = Math.max(24, Math.max(Math.abs(p[1] - fy), Math.abs(p[0] - fx) / ar));
  const at = h => { const w = h * ar; return { x: d.sx > 0 ? fx : fx - w, y: d.sy > 0 ? fy : fy - h, w, h }; };
  if (rectFits(V, at(want))) return at(want);
  let lo = 0, hi = want;
  for (let k = 0; k < 30; k++) { const m = (lo + hi) / 2; if (rectFits(V, at(m))) lo = m; else hi = m; }
  return lo > 8 ? at(lo) : d.start;
}

stage.addEventListener('pointermove', e => {
  if (!drag) return;
  const pos = evPos(e);
  if (drag.kind === 'pan') {
    const v = curVp();
    v.x = drag.v.x + pos[0] - drag.pos[0]; v.y = drag.v.y + pos[1] - drag.pos[1];
    draw();
  } else if (drag.kind === 'nudge') {
    const p = toSpace(pos), it = CUR();
    it.nudge = [drag.start[0] + p[0] - drag.p[0], drag.start[1] + p[1] - drag.p[1]];
    S.dirtyCrop = true; draw();
    lcd(`nudge ${it.nudge[0].toFixed(1)}, ${it.nudge[1].toFixed(1)} px`);
  } else if (drag.kind === 'crop-move' || drag.kind === 'crop-size') {
    const p = toSpace(pos), row = curRow();
    if (!row) return;
    const r = drag.kind === 'crop-move' ? moveCrop(drag.start, p[0] - drag.p[0], p[1] - drag.p[1]) : sizeCrop(drag, p);
    S.manual = { aspect: S.aspect, rect: r };
    row.rect = { ...r }; row.manual = true;
    draw();
  } else if (drag.kind === 'anchor' && !drag.raw && S.method === 'match') {
    if (Math.hypot(pos[0] - drag.pos[0], pos[1] - drag.pos[1]) > 6 * dpr) {
      const p = toSpace(pos);
      dragBox = { x: Math.min(p[0], drag.p[0]), y: Math.min(p[1], drag.p[1]), w: Math.abs(p[0] - drag.p[0]), h: Math.abs(p[1] - drag.p[1]) };
      draw();
    }
  }
});

stage.addEventListener('pointerup', e => {
  const d = drag; drag = null; stage.classList.remove('panning');
  if (!d) return;
  if (d.kind === 'nudge') { recompute(); refreshAll(); return; }
  if (d.kind === 'crop-move' || d.kind === 'crop-size') { recompute(); refreshAll(); draw(); return; }
  if (d.kind !== 'anchor') return;
  if (dragBox) {
    const b = dragBox; dragBox = null;
    if (b.w > 8 && b.h > 8) {
      S.roi = b;
      if (!S.anchor || !inBox(S.anchor, b)) S.anchor = [b.x + b.w / 2, b.y + b.h / 2];
      lcd('subject box set: only detail inside it is matched. press align all.');
      markStale();
    }
    refreshAll(); draw();
    return;
  }
  if (d.raw) {
    // as-shot view: a click marks the anchor in this photo
    const it = CUR(), [x, y] = d.p;
    if (x < 0 || y < 0 || x > it.W || y > it.H) return;
    if (S.cur === S.ref) S.anchor = [x, y];
    else it.manual = [x, y];
    if (S.method === 'manual') { manualFit(); advanceToUnmarked(); }
    else if (S.cur !== S.ref && it.map) {
      // pin the auto fit so the anchor lands exactly here
      it.map = pinAnchor(it.map, anchorPt(), [x, y]); it.res = { ...(it.res || {}), ok: true, status: 'pinned by hand' };
      recompute();
    }
    refreshAll(); draw();
    return;
  }
  // aligned view: set the anchor in the reference frame
  S.anchor = d.p;
  if (S.method === 'manual') { manualFit(); if (S.cur === S.ref) advanceToUnmarked(); }
  else if (S.kind === 'translate' && S.aligned) markStale('anchor moved: align again to re-centre on it.');
  recompute(); refreshAll(); draw();
});

const inBox = ([x, y], b) => x >= b.x && y >= b.y && x <= b.x + b.w && y <= b.y + b.h;

window.addEventListener('keydown', e => {
  if (e.target.closest && e.target.closest('input, textarea')) return;
  if (e.key === ' ') { spaceDown = true; if (!e.repeat) { e.preventDefault(); togglePlay(); } return; }
  if (e.ctrlKey || e.metaKey) return;
  const k = e.key.toLowerCase();
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'ArrowUp' || e.key === 'ArrowDown') {
    e.preventDefault();
    if (e.altKey && S.cur !== S.ref) {
      const it = CUR(), st = e.shiftKey ? 10 : 1, n = it.nudge || [0, 0];
      const dx = e.key === 'ArrowLeft' ? -st : e.key === 'ArrowRight' ? st : 0, dy = e.key === 'ArrowUp' ? -st : e.key === 'ArrowDown' ? st : 0;
      it.nudge = [n[0] + dx, n[1] + dy]; recompute(); refreshAll(); draw();
      return;
    }
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') select(S.cur - 1); else select(S.cur + 1);
    return;
  }
  if ('123456'.includes(k) && k.length === 1) { setView(['aligned', 'onion', 'diff', 'mean', 'crop', 'raw'][+k - 1]); return; }
  if (k === 'v') setTool('view');
  else if (k === 'a') setTool('anchor');
  else if (k === 'n') setTool('nudge');
  else if (k === 'c') setTool('frame');
  else if (k === 'f') { fitView(); draw(); }
  else if (k === 'x') toggleInclude();
  else if ((e.key === 'Delete' || e.key === 'Backspace') && e.shiftKey) clearAll();
  else if (e.key === 'Delete' || e.key === 'Backspace') removeCurrent();
});
window.addEventListener('keyup', e => { if (e.key === ' ') spaceDown = false; });

// ---------------------------------------------------------------- loading

async function addFiles(files) {
  const list = [...files].filter(f => (f.type || '').startsWith('image/') || /\.(jpe?g|png|webp|avif|gif|bmp|heic|heif|tiff?)$/i.test(f.name));
  if (!list.length) { toast('No photos in that drop.', true); return; }
  busy(true, 'loading photos', false);
  let bad = 0;
  try {
    for (let k = 0; k < list.length; k++) {
      const f = list[k];
      log(`${k + 1}/${list.length}  ${f.name}`); progress((k + 0.5) / list.length);
      try { S.items.push(await loadPhoto(f)); } catch (err) { console.warn(err); bad++; }
    }
  } finally { busy(false); }
  if (bad) toast(`${bad} file${bad > 1 ? 's' : ''} couldn't be decoded by this browser (HEIC needs Safari; or export as JPEG).`, true);
  if (S.items.length && !S.vp?.z) fitView();
  if (S.items.length === list.length - bad) { S.ref = pickReference(); S.cur = S.ref; fitView(); }
  markStale();
  lcd(`${S.items.length} photo${S.items.length === 1 ? '' : 's'}. ${S.anchor ? '' : 'click the anchor on the reference (or drag a box round the subject), then'} align all.`);
  renderStrip(); refreshAll(); draw();
}

async function loadPhoto(file) {
  const full = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const W = full.width, H = full.height;
  const k = Math.min(1, PREVIEW / Math.max(W, H));
  const prev = k < 1
    ? await createImageBitmap(full, { resizeWidth: Math.round(W * k), resizeHeight: Math.round(H * k), resizeQuality: 'high' })
    : full;
  const thumb = document.createElement('canvas');
  const tk = Math.max(168 / W, 112 / H);
  thumb.width = 168; thumb.height = 112;
  thumb.getContext('2d').drawImage(prev, (168 - W * tk) / 2, (112 - H * tk) / 2, W * tk, H * tk);
  if (prev !== full) full.close();
  return { id: nextId++, name: file.name, file, W, H, prev, tex: G.texture(prev), thumb, map: null, res: null, include: true, manual: null, nudge: null };
}

// default reference: the photo whose shape is most common, then the first of those
function pickReference() {
  if (S.anchor) return S.ref;
  return Math.min(S.ref, S.items.length - 1) || 0;
}

// luma at FINE long side, for the matcher
async function lumaOf(it, maxSide = FINE) {
  const full = await createImageBitmap(it.file, { imageOrientation: 'from-image' });
  const k = Math.min(1, maxSide / Math.max(full.width, full.height));
  const w = Math.max(1, Math.round(full.width * k)), h = Math.max(1, Math.round(full.height * k));
  const c = new OffscreenCanvas(w, h), ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(full, 0, 0, w, h);
  full.close();
  const l = lumaFromRGBA(ctx.getImageData(0, 0, w, h).data, w, h);
  return { data: l.data, w, h, W: it.W, H: it.H };
}

// ---------------------------------------------------------------- worker pool

let pool = [];
function makePool(n) {
  killPool();
  for (let k = 0; k < n; k++) {
    const w = new Worker(new URL('./core/stackworker.js', import.meta.url), { type: 'module' });
    const slot = { w, pending: new Map(), next: 1 };
    w.onmessage = ({ data }) => {
      const p = slot.pending.get(data.id); if (!p) return;
      slot.pending.delete(data.id);
      if (data.type === 'error') p.reject(new Error(data.message)); else p.resolve(data.result);
    };
    w.onerror = e => { for (const p of slot.pending.values()) p.reject(new Error(e.message || 'worker crashed')); slot.pending.clear(); };
    pool.push(slot);
  }
}
function killPool() {
  for (const s of pool) { s.w.terminate(); for (const p of s.pending.values()) p.reject(new Error('cancelled')); }
  pool = [];
}
function call(slot, kind, args, transfer = []) {
  return new Promise((resolve, reject) => {
    const id = slot.next++;
    slot.pending.set(id, { resolve, reject });
    slot.w.postMessage({ id, kind, args }, transfer);
  });
}

// ---------------------------------------------------------------- aligning

function markStale(msg) {
  if (S.method === 'manual') return;
  if (S.aligned && msg) lcd(msg);
  $('btn-align').classList.toggle('nudge', S.items.length > 1);
}

async function alignAll() {
  if (S.busy) return;
  if (S.items.length < 2) { toast('Add at least two photos.', true); return; }
  if (S.method === 'manual') { manualFit(); refreshAll(); draw(); return; }
  S.cancel = false;
  busy(true, S.method === 'bright' ? 'finding the light' : 'matching photos', true);
  const ref = R();
  const t0 = performance.now();
  try {
    if (S.method === 'bright') await alignBright();
    else await alignMatch();
    for (const it of S.items) it.nudge = null;
  } catch (err) {
    if (err.message !== 'cancelled') toast(err.message || String(err), true);
    else lcd('cancelled.');
    busy(false); killPool(); refreshAll(); draw();
    return;
  }
  killPool();
  busy(false);
  S.aligned = true;
  $('btn-align').classList.remove('nudge');
  recompute(); fitView();
  const ok = S.items.filter((it, i) => i !== S.ref && it.res?.ok).length, n = S.items.length - 1;
  const secs = ((performance.now() - t0) / 1000).toFixed(1);
  lcd(`${ok} of ${n} lined up on ${ref.name} in ${secs} s.${ok < n ? ' red = no match: fix by hand in "as shot" view or leave out.' : ''}`, ok < n);
  if (ok < n) toast(`${n - ok} photo${n - ok > 1 ? 's' : ''} didn't match. Mark the anchor by hand on ${n - ok > 1 ? 'them' : 'it'} (as-shot view, anchor tool) or leave ${n - ok > 1 ? 'them' : 'it'} out.`, true);
  renderStrip(); refreshAll(); draw();
}

async function alignMatch() {
  const ref = R();
  const n = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 4) - 1, S.items.length - 1));
  makePool(n);
  log('reading the reference');
  const refLum = await lumaOf(ref);
  const key = ref.id + ':' + Date.now();
  await Promise.all(pool.map(s => call(s, 'ref', { key, lum: refLum })));
  if (S.cancel) throw new Error('cancelled');
  const todo = S.items.map((_, i) => i).filter(i => i !== S.ref);
  let done = 0;
  const opts = { kind: S.kind, roi: S.roi, anchor: anchorPt(), key };
  progress(0.05);
  const run = async slot => {
    while (todo.length) {
      if (S.cancel) throw new Error('cancelled');
      const i = todo.shift(), it = S.items[i];
      const lum = await lumaOf(it);
      const r = await call(slot, 'pair', { ...opts, lum }, [lum.data.buffer]);
      it.map = r.ok ? r.H : null;
      it.res = { ok: r.ok, status: r.ok ? 'matched' : (r.reason || 'no match'), inliers: r.inliers, rms: r.rms };
      done++;
      progress(0.05 + 0.85 * done / (S.items.length - 1));
      log(`${done}/${S.items.length - 1}  ${it.name}: ${it.res.status}${r.ok ? ` (${r.inliers} points)` : ''}`);
      renderStrip();
    }
  };
  await Promise.all(pool.map(run));
  // second chance for the misses: line them up through a photo that matched
  const missed = S.items.map((it, i) => (i !== S.ref && !it.res?.ok ? i : -1)).filter(i => i >= 0);
  const good = S.items.map((it, i) => (i !== S.ref && it.res?.ok ? i : -1)).filter(i => i >= 0);
  for (const i of missed) {
    if (!good.length) break;
    log(`retrying ${S.items[i].name} via the others`);
    const it = S.items[i], lum = await lumaOf(it);
    let best = null;
    for (const j of good.slice(0, 4)) {
      if (S.cancel) throw new Error('cancelled');
      const base = await lumaOf(S.items[j]);
      let roiJ = null;
      if (S.roi) {
        const q = [[S.roi.x, S.roi.y], [S.roi.x + S.roi.w, S.roi.y], [S.roi.x + S.roi.w, S.roi.y + S.roi.h], [S.roi.x, S.roi.y + S.roi.h]].map(([x, y]) => apply(S.items[j].map, x, y));
        const xs = q.map(p => p[0]), ys = q.map(p => p[1]);
        roiJ = { x: Math.min(...xs), y: Math.min(...ys), w: Math.max(...xs) - Math.min(...xs), h: Math.max(...ys) - Math.min(...ys) };
      }
      const r = await call(pool[0], 'pair', { kind: S.kind === 'translate' ? 'similarity' : S.kind, roi: roiJ, lum: { ...lum, data: lum.data.slice() }, base });
      if (r.ok && (!best || r.inliers > best.r.inliers)) best = { r, j };
    }
    if (best) {
      let H = mul3(best.r.H, S.items[best.j].map);
      if (S.kind === 'translate') H = reduceToTranslation(H, anchorPt());
      it.map = H;
      it.res = { ok: true, status: `matched via #${best.j + 1}`, inliers: best.r.inliers, rms: best.r.rms };
    }
  }
  for (const it of S.items) {
    if (it.map && !sane(inv3(it.map), it.W, it.H)) { it.map = null; it.res = { ok: false, status: 'implausible fit' }; }
  }
}

async function alignBright() {
  const ref = R();
  const spot = async it => {
    const l = await lumaOf(it, 800);
    return brightestSpot({ data: l.data, w: l.w, h: l.h }, it.W, it.H);
  };
  const a = await spot(ref);
  if (!a) throw new Error('No bright spot in the reference.');
  S.anchor = [a.x, a.y];
  let k = 0;
  for (const [i, it] of S.items.entries()) {
    if (S.cancel) throw new Error('cancelled');
    progress(++k / S.items.length);
    if (i === S.ref) continue;
    log(`${it.name}`);
    const b = await spot(it), sx = it.W / ref.W;
    it.map = b ? mul3(translation(b.x - a.x * sx, b.y - a.y * sx), scaling(sx)) : null;
    it.res = b ? { ok: true, status: 'light found' } : { ok: false, status: 'no bright spot' };
    renderStrip();
  }
}

// by-hand mode: each photo's clicked point goes onto the reference anchor
function manualFit() {
  const ref = R(); if (!ref) return;
  const a = anchorPt();
  if (!S.anchor) S.anchor = a;
  S.items.forEach((it, i) => {
    if (i === S.ref) return;
    const sx = it.W / ref.W;
    if (it.manual) {
      it.map = mul3(translation(it.manual[0] - a[0] * sx, it.manual[1] - a[1] * sx), scaling(sx));
      it.res = { ok: true, status: 'by hand' };
    } else if (!it.res?.ok || it.res.status === 'by hand') {
      it.map = null; it.res = { ok: false, status: 'click its anchor' };
    }
  });
  S.aligned = true;
  recompute(); renderStrip();
  const left = S.items.filter((it, i) => i !== S.ref && !it.res?.ok).length;
  lcd(left ? `${left} photo${left > 1 ? 's' : ''} still need the anchor clicked.` : 'every photo marked. check "mean" or flick through with space.');
}

function advanceToUnmarked() {
  const n = S.items.length;
  for (let k = 1; k < n; k++) {
    const i = (S.cur + k) % n;
    if (i !== S.ref && !S.items[i].manual) { select(i); return; }
  }
}

// ---------------------------------------------------------------- reference & set edits

function setReference(i) {
  if (i === S.ref || !S.items[i]) return;
  const old = S.ref, M = effMap(i); // old ref -> new ref
  const Mi = inv3(M);
  if (S.aligned && S.items[i].map && Mi) {
    // keep every fit: new ref -> j = (old ref -> j) o (new ref -> old ref)
    S.items.forEach((it, j) => {
      if (j === old) { it.map = Mi; it.res = { ok: true, status: 'matched' }; }
      else if (j !== i && it.map) it.map = mul3(effMap(j), Mi);
      it.nudge = null;
    });
    if (S.anchor) S.anchor = apply(M, ...S.anchor);
    if (S.roi) {
      const q = [[S.roi.x, S.roi.y], [S.roi.x + S.roi.w, S.roi.y + S.roi.h]].map(([x, y]) => apply(M, x, y));
      S.roi = { x: Math.min(q[0][0], q[1][0]), y: Math.min(q[0][1], q[1][1]), w: Math.abs(q[1][0] - q[0][0]), h: Math.abs(q[1][1] - q[0][1]) };
    }
  } else {
    S.items.forEach(it => { it.map = null; it.res = null; it.nudge = null; });
    S.anchor = null; S.roi = null; S.aligned = false;
  }
  S.items[i].map = null; S.items[i].res = { ok: true, status: 'reference' };
  S.ref = i;
  recompute(); fitView(); renderStrip(); refreshAll(); draw();
  lcd(`${S.items[i].name} is now the frame everything lines up to.`);
}

function toggleInclude() {
  const it = CUR(); if (!it) return;
  if (S.cur === S.ref) { toast('The reference is always in. Make another photo the reference first.'); return; }
  it.include = !it.include;
  recompute(); renderStrip(); refreshAll(); draw();
}

function removeCurrent() { removeAt(S.cur); }

function removeAt(i) {
  const it = S.items[i];
  if (!it || S.busy) return;
  G.free(it.tex); it.prev?.close?.();
  const wasRef = i === S.ref;
  S.items.splice(i, 1);
  if (!S.items.length) { resetSet(); return; }
  if (wasRef) {
    S.ref = 0; S.items.forEach(x => { x.map = null; x.res = null; x.nudge = null; }); S.aligned = false; S.anchor = null; S.roi = null;
    lcd(`${it.name} removed. it was the reference, so ${S.items[0].name} is now: align again.`);
  } else {
    if (i < S.ref) S.ref--;
    lcd(`${it.name} removed. ${S.items.length} photo${S.items.length > 1 ? 's' : ''} left.`);
  }
  if (i < S.cur || S.cur >= S.items.length) S.cur = Math.max(0, S.cur - 1);
  S.items[S.ref].res = { ok: true, status: 'reference' };
  recompute(); fitView(); renderStrip(); refreshAll(); draw();
}

// empty the set: back to the drop screen, settings kept
function resetSet() {
  for (const it of S.items) { G.free(it.tex); it.prev?.close?.(); }
  if (S.playing) togglePlay();
  Object.assign(S, { items: [], ref: 0, cur: 0, anchor: null, roi: null, aligned: false, report: null, costs: [] });
  armClear(false);
  $('out-line').textContent = '\u2013';
  recompute(); fitView(); renderStrip(); refreshAll(); draw();
  lcd('cleared. drop a set of photos to begin.');
}

// "clear all" asks once: the first press arms it for a few seconds
let clearTimer = 0;
function armClear(on) {
  clearTimeout(clearTimer);
  const b = $('btn-clear');
  b.dataset.armed = on ? '1' : '';
  b.textContent = on ? 'sure? clear' : 'clear all';
  b.classList.toggle('nudge', on);
  if (on) clearTimer = setTimeout(() => armClear(false), 3000);
}
function clearAll() {
  if (S.busy || !S.items.length) return;
  if ($('btn-clear').dataset.armed) resetSet(); else { armClear(true); lcd(`press clear all again to remove all ${S.items.length} photos.`); }
}

function select(i) {
  if (!S.items.length) return;
  S.cur = (i + S.items.length) % S.items.length;
  const c = CUR();
  S.rawVp = fitTo(0, 0, c.W, c.H);
  renderStrip(); refreshAll(); draw();
  const el = $('film').children[S.cur]; el?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
}

// ---------------------------------------------------------------- crop

function recompute() {
  const ref = R();
  if (!ref || !S.aligned) { S.report = null; S.costs = []; renderAspects(); return; }
  const maps = S.items.map((_, i) => (usable(i) ? effMap(i) : null));
  const inc = S.items.map((_, i) => usable(i));
  S.report = cropReport(S.items, maps, inc, anchorPt(), { place: S.place, zoom: S.zoom });
  applyManual(maps, inc);
  const r = ASPECTS.find(a => a.id === S.aspect).r;
  S.costs = S.items.length <= 60 ? leaveOneOut(S.items, maps, inc, anchorPt(), r, S.place) : [];
  renderAspects(); renderStrip();
  const row = S.report.rows.find(x => x.id === S.aspect);
  const n = inc.filter(Boolean).length;
  if (!row?.rect) {
    $('crop-note').textContent = S.report.region.length < 3
      ? 'the photos don’t share any common area. leave the odd ones out.'
      : 'no frame of this shape fits around the anchor in every photo. try "biggest", another shape, or leave a photo out.';
  } else {
    const [w, h] = outSize(row);
    const pct = Math.round((row.refW * row.refH) / (ref.W * ref.H) * 100);
    $('crop-note').textContent = `${n} photos share this ${row.id} frame: ${w}×${h} px, ${pct}% of the reference.${row.manual ? ' placed by hand.' : ''}`;
  }
}

// a hand-placed crop replaces the automatic one for its shape while it still fits
const rectFits = (V, r) => V.length >= 3 && [[r.x, r.y], [r.x + r.w, r.y], [r.x + r.w, r.y + r.h], [r.x, r.y + r.h]].every(([x, y]) => inside(V, x, y));
function applyManual(maps, inc) {
  const m = S.manual;
  if (!m || m.aspect !== S.aspect || !S.report) return;
  if (!rectFits(S.report.region, m.rect)) { S.manual = null; lcd('the hand-placed crop no longer fits every photo: back to automatic.'); return; }
  const row = S.report.rows.find(x => x.id === S.aspect);
  const k = nativeScale(S.items, maps, inc, m.rect);
  Object.assign(row, { rect: { ...m.rect }, refW: m.rect.w, refH: m.rect.h, outW: Math.floor(m.rect.w * k), outH: Math.floor(m.rect.h * k), manual: true });
}

function outSize(row) {
  if (!row?.rect) return [0, 0];
  const r = row.rect.w / row.rect.h;
  let w, h;
  if (S.size === 'native') [w, h] = [row.outW, row.outH];
  else { const s = +S.size; [w, h] = r >= 1 ? [Math.round(s * r), s] : [s, Math.round(s / r)]; }
  if (S.fmt === 'mp4') {
    // H.264 wants even sizes, and most encoders stop at 4K
    const k = Math.min(1, 3840 / Math.max(w, h));
    w = Math.max(2, Math.floor(w * k / 2) * 2); h = Math.max(2, Math.floor(h * k / 2) * 2);
  }
  return [w, h];
}

function renderAspects() {
  const box = $('aspects');
  if (!box.children.length) {
    for (const a of ASPECTS) {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'aspect'; b.dataset.id = a.id; b.setAttribute('role', 'listitem');
      const sw = a.r >= 1 ? 16 : Math.round(16 * a.r), sh = a.r >= 1 ? Math.round(16 / a.r) : 16;
      b.innerHTML = `<span class="shape"><i style="width:${sw}px;height:${sh}px"></i></span><b>${a.id}</b><small>&ndash;</small>`;
      b.addEventListener('click', () => { S.aspect = a.id; S.manual = null; recompute(); refreshAll(); if (S.view === 'crop') fitView(); draw(); });
      box.append(b);
    }
  }
  for (const b of box.children) {
    const row = S.report?.rows.find(r => r.id === b.dataset.id);
    b.setAttribute('aria-pressed', String(b.dataset.id === S.aspect));
    const sm = b.querySelector('small');
    b.classList.toggle('none', !!S.report && !row?.rect);
    sm.textContent = !S.report ? '–' : row?.rect ? outSize(row).join('×') : 'won’t fit';
  }
}

// ---------------------------------------------------------------- film strip

function renderStrip() {
  const film = $('film');
  const ids = new Set(S.items.map(it => String(it.id)));
  for (const el of [...film.children]) if (!ids.has(el.dataset.id)) el.remove();
  S.items.forEach((it, i) => {
    let el = film.children[i];
    if (!el || el.dataset.id !== String(it.id)) {
      el = document.createElement('div');
      el.className = 'shot'; el.setAttribute('role', 'option'); el.tabIndex = 0; el.dataset.id = it.id;
      el.innerHTML = '<span class="num"></span><span class="ref" hidden>REF</span><span class="cost" hidden></span><div class="meta"><span class="led"></span><span class="nm"></span><button type="button" class="x" aria-label="Remove this photo" title="Remove this photo">\u00d7</button></div>';
      el.prepend(it.thumb);
      const idx = () => S.items.findIndex(x => x.id === it.id);
      el.addEventListener('click', () => select(idx()));
      el.addEventListener('keydown', e => { if (e.key === 'Enter') select(idx()); });
      el.querySelector('.x').addEventListener('click', e => { e.stopPropagation(); removeAt(idx()); });
      el.addEventListener('dblclick', () => setReference(S.items.findIndex(x => x.id === it.id)));
      if (film.children[i]) film.insertBefore(el, film.children[i]); else film.append(el);
    }
    el.setAttribute('aria-selected', String(i === S.cur));
    el.classList.toggle('out', !it.include);
    el.querySelector('.num').textContent = i + 1;
    el.querySelector('.ref').hidden = i !== S.ref;
    const led = el.querySelector('.led');
    led.className = 'led' + (i === S.ref ? ' p' : it.res?.ok ? ' g' : it.res ? ' r' : '');
    const st = i === S.ref ? 'reference' : !it.include ? 'left out' : it.res ? it.res.status : it.name;
    el.querySelector('.nm').textContent = st;
    el.title = `${it.name}\n${it.W}×${it.H}${it.res ? '\n' + it.res.status : ''}${it.res?.inliers ? ` (${it.res.inliers} points${it.res.rms != null ? `, ${it.res.rms.toFixed(2)} px rms` : ''})` : ''}\ndouble-click: make reference`;
    const cost = el.querySelector('.cost'), c = S.costs[i];
    cost.hidden = !(c && c > 1.1 && it.include && i !== S.ref);
    if (!cost.hidden) { cost.textContent = `+${Math.round((c - 1) * 100)}%`; cost.title = `Leaving this photo out makes the ${S.aspect} crop ${Math.round((c - 1) * 100)}% bigger`; }
  });
}

// ---------------------------------------------------------------- export

async function fullTexture(GL, it) {
  const full = await createImageBitmap(it.file, { imageOrientation: 'from-image' });
  let src = full;
  if (Math.max(full.width, full.height) > GL.maxTex) {
    const k = GL.maxTex / Math.max(full.width, full.height);
    src = await createImageBitmap(full, { resizeWidth: Math.floor(full.width * k), resizeHeight: Math.floor(full.height * k), resizeQuality: 'high' });
    full.close();
  }
  const tex = GL.texture(src);
  src.close();
  return tex;
}

let XG = null;
const exportGL = () => (XG ??= new StackGL(document.createElement('canvas'), { preserve: false }));

async function renderCrop(i, row, [w, h]) {
  const GL = exportGL(), it = S.items[i];
  const tex = await fullTexture(GL, it);
  try {
    const z = w / row.rect.w;
    const px = GL.renderToPixels({ tex, map: effMap(i), w: it.W, h: it.H }, { x: -row.rect.x * z, y: -row.rect.y * z, z }, w, h);
    const c = new OffscreenCanvas(w, h);
    c.getContext('2d').putImageData(new ImageData(px, w, h), 0, 0);
    return await c.convertToBlob(S.fmt === 'png' ? { type: 'image/png' } : { type: 'image/jpeg', quality: 0.93 });
  } finally { GL.free(tex); }
}

const baseName = n => n.replace(/\.[^.]+$/, '');
const fileName = (it, row) => `${baseName(it.name)}_${row.id.replace(':', 'x')}.${S.fmt === 'png' ? 'png' : 'jpg'}`;

async function exportAll(onlyCurrent = false) {
  if (S.busy) return;
  if (S.fmt === 'mp4' && !onlyCurrent) return exportVideo();
  const row = S.report?.rows.find(r => r.id === S.aspect);
  if (!S.aligned || !row?.rect) { toast('Align first, and pick a frame shape that fits.', true); return; }
  const ids = onlyCurrent ? [S.cur] : S.items.map((_, i) => i).filter(usable);
  if (onlyCurrent && !usable(S.cur)) { toast('This photo isn’t lined up (or is left out).', true); return; }
  const size = outSize(row);
  prepareDownloads();
  S.cancel = false;
  busy(true, `exporting ${row.id}`, true);
  try {
    if (onlyCurrent) {
      log(S.items[S.cur].name);
      const blob = await renderCrop(S.cur, row, size);
      busy(false);
      const r = await saveFile(fileName(S.items[S.cur], row), blob);
      $('out-line').textContent = r === 'saved' ? `saved ${fileName(S.items[S.cur], row)}` : 'save declined.';
      return;
    }
    const zip = new ZipWriter();
    for (let k = 0; k < ids.length; k++) {
      if (S.cancel) throw new Error('cancelled');
      const it = S.items[ids[k]];
      log(`${k + 1}/${ids.length}  ${it.name}`); progress((k + 0.2) / ids.length);
      const name = `${String(k + 1).padStart(2, '0')}_${fileName(it, row)}`;
      await zip.add(name, await renderCrop(ids[k], row, size));
    }
    await zip.add('align.json', new Blob([JSON.stringify(fitJSON(), null, 1)], { type: 'application/json' }));
    busy(false);
    const zname = `align_${row.id.replace(':', 'x')}_${size.join('x')}.zip`;
    const r = await saveFile(zname, zip.blob());
    $('out-line').textContent = r === 'saved' ? `saved ${ids.length} crops, ${size.join('×')}` : 'save declined.';
  } catch (err) {
    busy(false);
    if (err.message === 'cancelled') lcd('export cancelled.'); else toast(err.message || String(err), true);
  }
}

// every included photo, in strip order, held for S.hold frames at S.fps
async function exportVideo() {
  const row = curRow();
  if (!row?.rect) { toast('Align first, and pick a frame shape that fits.', true); return; }
  const ids = S.items.map((_, i) => i).filter(usable);
  if (ids.length < 1) return;
  const [w, h] = outSize(row), fps = S.fps, hold = S.hold;
  prepareDownloads();
  S.cancel = false;
  busy(true, `video ${row.id}`, true);
  let GL = null;
  try {
    log('loading the video encoder');
    const M = await mb();
    const codec = await M.getFirstEncodableVideoCodec(['avc', 'vp9', 'av1', 'hevc'], { width: w, height: h, frameRate: fps });
    if (!codec) throw new Error(`This browser can't encode ${w}\u00d7${h} video. Pick 1080, or use Chrome or Edge on a desktop.`);
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    GL = new StackGL(canvas, { preserve: true, alpha: false });
    const output = new M.Output({ format: new M.Mp4OutputFormat({ fastStart: 'in-memory' }), target: new M.BufferTarget() });
    const bitrate = Math.round(Math.min(80e6, Math.max(6e6, w * h * fps * 0.12)));
    const src = new M.CanvasSource(canvas, { codec, bitrate });
    output.addVideoTrack(src, { frameRate: fps });
    await output.start();
    const z = w / row.rect.w, view = { x: -row.rect.x * z, y: -row.rect.y * z, z };
    let frame = 0;
    for (let k = 0; k < ids.length; k++) {
      if (S.cancel) { await output.cancel(); throw new Error('cancelled'); }
      const i = ids[k], it = S.items[i];
      log(`${codec.toUpperCase()} \u00b7 photo ${k + 1} of ${ids.length} \u00b7 ${it.name}`);
      progress((k + 0.5) / ids.length);
      const tex = await fullTexture(GL, it);
      try {
        GL.begin(w, h);
        GL.draw({ tex, map: effMap(i), w: it.W, h: it.H }, view);
        // constant frame rate: the same picture repeated, one sample per frame
        for (let f = 0; f < hold; f++, frame++) await src.add(frame / fps, 1 / fps);
      } finally { GL.free(tex); }
    }
    log('finishing the file');
    await output.finalize();
    const blob = new Blob([output.target.buffer], { type: 'video/mp4' });
    busy(false);
    const name = `align_${row.id.replace(':', 'x')}_${w}x${h}_${fps}fps.mp4`;
    const r = await saveFile(name, blob);
    $('out-line').textContent = r === 'saved' ? `saved ${name}, ${(blob.size / 1048576).toFixed(1)} MB` : 'save declined.';
    if (r === 'saved') lcd(`video: ${ids.length} photos, ${frame} frames, ${(frame / fps).toFixed(2)} s at ${fps} fps, ${w}\u00d7${h}.`);
  } catch (err) {
    busy(false);
    if (err.message === 'cancelled') lcd('export cancelled.'); else toast(err.message || String(err), true);
  } finally { GL?.release(); }
}

function fitJSON() {
  const row = S.report?.rows.find(r => r.id === S.aspect);
  return {
    app: 'align-stack', version: 1,
    note: 'map: 3x3 row-major, reference pixels -> this photo’s pixels (full resolution, as displayed after EXIF rotation)',
    reference: R()?.name, anchor: anchorPt(), subject: S.roi, method: S.method, kind: S.kind,
    crop: row?.rect ? { aspect: row.id, rect: row.rect, size: outSize(row), byHand: !!row.manual } : null,
    photos: S.items.map((it, i) => ({ name: it.name, w: it.W, h: it.H, included: usable(i), status: it.res?.status || null, map: usable(i) ? effMap(i) : null })),
  };
}

// ---------------------------------------------------------------- flick playback

let playTimer = 0;
// flick at the video's timing when exporting mp4, otherwise 6 photos a second
const playMs = () => (S.fmt === 'mp4' ? (1000 * S.hold) / S.fps : 1000 / 6);
function retime() { if (S.playing) { togglePlay(); togglePlay(); } }
function togglePlay() {
  S.playing = !S.playing;
  $('play-icon').setAttribute('d', S.playing ? 'M7 5h4v14H7zM13 5h4v14h-4z' : 'M8 5v14l11-7z');
  clearInterval(playTimer);
  if (S.playing) {
    if (S.view === 'diff' || S.view === 'mean' || S.view === 'onion') setView('aligned');
    playTimer = setInterval(() => {
      const n = S.items.length; if (n < 2) return;
      for (let k = 1; k <= n; k++) {
        const i = (S.cur + k) % n;
        if (usable(i) || !S.aligned) { S.cur = i; break; }
      }
      renderStrip(); draw();
    }, playMs());
  }
}

// ---------------------------------------------------------------- panels

function setTool(t) { S.tool = t; stage.dataset.tool = t; refreshAll(); updateZoom(); draw(); }
function setView(v) {
  const wasCrop = S.view === 'crop';
  S.view = v; if (v === 'crop' || wasCrop) fitView();
  refreshAll(); updateZoom(); draw();
}

const notes = {
  match: 'click the anchor on the reference. drag a box round the subject to match only that (ignores sky, cranes, people).',
  bright: 'finds the brightest spot in each photo (the sun) and puts it on the same point. no clicking needed.',
  manual: 'click the same point in every photo: the reference first, then each one (it jumps to the next). fine for faces.',
};

function refreshAll() {
  savePrefs();
  const has = S.items.length > 0, two = S.items.length > 1;
  for (const b of document.querySelectorAll('#views button')) b.setAttribute('aria-pressed', String(b.dataset.view === S.view));
  for (const b of document.querySelectorAll('#method .chip')) b.setAttribute('aria-pressed', String(b.dataset.method === S.method));
  for (const b of document.querySelectorAll('#kind .chip')) b.setAttribute('aria-pressed', String(b.dataset.kind === S.kind));
  for (const b of document.querySelectorAll('#tools .chip')) b.setAttribute('aria-pressed', String(b.dataset.tool === S.tool));
  for (const b of document.querySelectorAll('#place .chip')) b.setAttribute('aria-pressed', String(b.dataset.place === S.place));
  for (const b of document.querySelectorAll('#size .chip')) b.setAttribute('aria-pressed', String(b.dataset.size === S.size));
  for (const b of document.querySelectorAll('#fmt .chip')) b.setAttribute('aria-pressed', String(b.dataset.fmt === S.fmt));
  for (const b of document.querySelectorAll('#fps .chip')) b.setAttribute('aria-pressed', String(+b.dataset.fps === S.fps));
  const vid = S.fmt === 'mp4', nUse = S.items.filter((_, i) => usable(i)).length;
  $('video-opts').hidden = !vid;
  $('hold-val').textContent = `${S.hold} frame${S.hold > 1 ? 's' : ''} \u00b7 ${(S.hold / S.fps).toFixed(2)} s`;
  $('video-note').textContent = nUse ? `${nUse} photos \u00d7 ${S.hold} = ${nUse * S.hold} frames, ${(nUse * S.hold / S.fps).toFixed(2)} s. order = the strip. space previews it at this speed.` : '';
  $('btn-auto-crop').hidden = !(S.manual && S.manual.aspect === S.aspect);
  $('kind-block').hidden = S.method !== 'match';
  $('anchor-note').textContent = notes[S.method];
  $('btn-clear-roi').hidden = !(S.roi && S.method === 'match');
  $('drop').classList.toggle('loaded', has);
  $('drop-label').textContent = has ? `${S.items.length} photo${S.items.length > 1 ? 's' : ''} · add more` : 'add photos';
  $('ref-name').textContent = has ? `${R().name}  ${R().W}×${R().H}` : 'no reference yet';
  $('btn-make-ref').hidden = !has || S.cur === S.ref;
  $('btn-toggle').disabled = !has || S.cur === S.ref;
  $('btn-toggle').textContent = CUR() && !CUR().include ? 'put back in' : 'leave out';
  $('btn-remove').disabled = !has;
  $('btn-clear').disabled = !has || S.busy;
  $('btn-align').disabled = !two || S.busy;
  $('btn-align').innerHTML = `<span aria-hidden="true">&#x25B6;</span> ${S.method === 'manual' ? 'apply clicks' : 'align all'}`;
  $('btn-reset-nudge').hidden = !(CUR()?.nudge);
  const ready = S.aligned && S.report?.rows.find(r => r.id === S.aspect)?.rect;
  $('btn-export').disabled = !ready; $('btn-export-one').disabled = !ready || !usable(S.cur);
  $('btn-save-json').disabled = !S.aligned;
  $('btn-export').textContent = vid ? 'save video (.mp4)' : ready ? `save ${nUse} crops (.zip)` : 'save all crops (.zip)';
  $('btn-export-one').textContent = vid ? 'save this frame' : 'save this one';
  $('zoom-val').textContent = `${Math.round((1 - S.zoom) * 100)}%`;
  $('led-ready').className = 'led' + (S.aligned ? ' g' : '');
  const it = CUR();
  if (it && S.aligned) {
    const r = it.res;
    $('align-line').textContent = S.cur === S.ref ? `#${S.cur + 1} reference` :
      `#${S.cur + 1} ${r?.status || '–'}${r?.inliers ? ` · ${r.inliers} pts` : ''}${r?.rms != null ? ` · ${r.rms.toFixed(2)} px` : ''}${it.nudge ? ' · nudged' : ''}`;
  } else $('align-line').textContent = two ? 'ready' : '–';
}

// ---------------------------------------------------------------- busy / toast

function busy(on, label = '', cancellable = false) {
  S.busy = on;
  $('busy').hidden = !on;
  $('led-busy').className = 'led' + (on ? ' p pulse' : '');
  if (on) { $('busy-label').textContent = label; log(''); progress(0.02); $('busy-cancel').hidden = !cancellable; }
  refreshAll();
}
const log = t => { $('busy-stage').textContent = t; };
const progress = f => { $('busy-bar').style.width = `${Math.round(Math.max(0, Math.min(1, f)) * 100)}%`; };
function lcd(msg, err = false) { const el = $('lcd'); el.textContent = msg; el.classList.toggle('err', err); }
function toast(msg, error = false) {
  const box = $('toasts'), t = document.createElement('div');
  t.className = 'toast' + (error ? ' err' : '');
  t.textContent = msg;
  box.append(t);
  while (box.children.length > 3) box.firstChild.remove();
  setTimeout(() => t.remove(), error ? 9000 : 5000);
}
window.addEventListener('unhandledrejection', e => { console.error(e.reason); toast(e.reason?.message || String(e.reason), true); });

// ---------------------------------------------------------------- wiring

const chipGroup = (id, attr, fn) => {
  for (const b of document.querySelectorAll(`#${id} [data-${attr}]`)) b.addEventListener('click', () => fn(b.dataset[attr]));
};
chipGroup('views', 'view', setView);
chipGroup('tools', 'tool', setTool);
chipGroup('method', 'method', m => {
  if (S.method === m) return;
  S.method = m;
  if (m === 'manual') { setTool('anchor'); manualFit(); }
  else if (S.aligned) markStale('method changed: press align all.');
  refreshAll(); draw();
});
chipGroup('kind', 'kind', k => { if (S.kind !== k) { S.kind = k; markStale('model changed: press align all.'); refreshAll(); } });
chipGroup('place', 'place', p => { S.place = p; S.manual = null; recompute(); refreshAll(); if (S.view === 'crop') fitView(); draw(); });
chipGroup('size', 'size', s => { S.size = s; recompute(); refreshAll(); draw(); });
chipGroup('fmt', 'fmt', f => { S.fmt = f; recompute(); refreshAll(); retime(); draw(); });
chipGroup('fps', 'fps', f => { S.fps = +f; refreshAll(); retime(); });
$('hold').value = S.hold;
$('hold').addEventListener('input', () => { S.hold = +$('hold').value; refreshAll(); retime(); });
$('btn-auto-crop').addEventListener('click', () => { S.manual = null; recompute(); refreshAll(); if (S.view === 'crop') fitView(); draw(); });

function paintFader(el) {
  const min = Number(el.min), max = Number(el.max), v = Number(el.value);
  el.style.setProperty('--pct', `${((v - min) / (max - min || 1)) * 100}%`);
}
$('zoom').addEventListener('input', () => {
  paintFader($('zoom')); S.zoom = 1 - Number($('zoom').value) / 100; S.manual = null;
  recompute(); refreshAll(); if (S.view === 'crop') fitView(); draw();
});
paintFader($('zoom'));

$('drop').addEventListener('click', () => $('file-input').click());
$('file-input').addEventListener('change', e => { if (e.target.files.length) addFiles(e.target.files); e.target.value = ''; });
for (const el of [stage, $('drop'), document.body]) {
  el.addEventListener('dragover', e => { e.preventDefault(); (el === document.body ? stage : el).classList.add('drag'); });
  el.addEventListener('dragleave', () => { stage.classList.remove('drag'); $('drop').classList.remove('drag'); });
}
document.body.addEventListener('drop', e => {
  e.preventDefault(); stage.classList.remove('drag'); $('drop').classList.remove('drag');
  if (e.dataTransfer?.files?.length && !S.busy) addFiles(e.dataTransfer.files);
});

$('btn-align').addEventListener('click', alignAll);
$('btn-make-ref').addEventListener('click', () => setReference(S.cur));
$('btn-toggle').addEventListener('click', toggleInclude);
$('btn-remove').addEventListener('click', removeCurrent);
$('btn-clear').addEventListener('click', clearAll);
$('btn-clear-roi').addEventListener('click', () => { S.roi = null; markStale('subject box cleared: press align all.'); refreshAll(); draw(); });
$('btn-reset-nudge').addEventListener('click', () => { if (CUR()) { CUR().nudge = null; recompute(); refreshAll(); draw(); } });
$('btn-export').addEventListener('click', () => exportAll(false));
$('btn-export-one').addEventListener('click', () => exportAll(true));
$('btn-save-json').addEventListener('click', async () => {
  const r = await saveFile('align.json', new Blob([JSON.stringify(fitJSON(), null, 1)], { type: 'application/json' }));
  $('out-line').textContent = r === 'saved' ? 'saved align.json' : 'save declined.';
});
$('busy-cancel').addEventListener('click', () => { S.cancel = true; killPool(); });
$('fit').addEventListener('click', () => { fitView(); draw(); });
$('theme').addEventListener('click', () => setTheme(document.body.dataset.ui === 'light' ? 'dark' : 'light'));
$('play').addEventListener('click', togglePlay);

renderAspects(); refreshAll();
// test hook: lets a harness feed files without a file picker
window.__align = { addFiles, alignAll, S, setReference, recompute, refreshAll, draw };
