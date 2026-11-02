// Stack mode engine: line up a set of photos on one anchor (a building, the sun,
// a face, any point), then work out how much of the frame survives in every shot.
//
// All maps are "reference -> image": H_i takes reference-image pixels (full res) to
// image i's pixels (full res). Rendering image i into the reference frame samples it
// at H_i(p). DOM-free; runs in a worker and in Node.

import { detectAndDescribe, matchFeatures } from './features.js';
import { blur, downscale } from './image.js';
import {
  I3, mul3, inv3, apply, translation, scaling, ransac, fitModel, fitSimilarity, sane,
  residuals, quadOf, intersectAll, bestRect, zoomRect, localScale, polyArea,
} from './geom.js';

export const WORK = 1024;     // long side used for feature detection
export const FINE = 2048;     // long side used for sub-pixel refinement

// ---------------------------------------------------------------- per-image prep
// lum: gray image at <= FINE long side; W/H: full-res size.
export function prepare(lum, W, H, { mask = null, count = 2000 } = {}) {
  const ks = Math.min(1, WORK / Math.max(lum.w, lum.h));
  const work = ks < 1 ? downscale(lum, Math.round(lum.w * ks), Math.round(lum.h * ks)) : lum;
  const toWork = work.w / W;  // full px -> work px
  const m = mask ? (x, y) => mask(x / toWork, y / toWork) : null;
  const feats = detectAndDescribe(work, { count, mask: m });
  return { W, H, fine: lum, work, toWork, toFine: lum.w / W, feats };
}

// ---------------------------------------------------------------- NCC refinement

function sample(img, x, y) {
  const { data, w, h } = img;
  if (x < 0 || y < 0 || x > w - 1.001 || y > h - 1.001) return NaN;
  const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i = y0 * w + x0;
  return (data[i] * (1 - fx) + data[i + 1] * fx) * (1 - fy) + (data[i + w] * (1 - fx) + data[i + w + 1] * fx) * fy;
}

// For reference point p (fine px), compare its patch with the target sampled through
// the current map G (ref fine px -> target fine px) over a +-S window. Returns the
// sub-pixel shift d (ref fine px) so that ref(p) ~ target(G(p + d)), and the score.
function nccShift(R, T, G, px, py, r, S) {
  const n = 2 * r + 1, N = n * n, span = 2 * (r + S) + 1;
  const P = new Float32Array(N);
  let pm = 0;
  for (let j = -r, k = 0; j <= r; j++) for (let i = -r; i <= r; i++, k++) {
    const v = sample(R, px + i, py + j);
    if (v !== v) return null;
    P[k] = v; pm += v;
  }
  pm /= N;
  let pv = 0;
  for (let k = 0; k < N; k++) { P[k] -= pm; pv += P[k] * P[k]; }
  if (pv < N * 2e-5) return null;  // flat patch
  const Wd = new Float32Array(span * span);
  for (let j = 0; j < span; j++) for (let i = 0; i < span; i++) {
    const [u, v] = apply(G, px - r - S + i, py - r - S + j);
    const s = sample(T, u, v);
    if (s !== s) return null;
    Wd[j * span + i] = s;
  }
  const sc = new Float32Array((2 * S + 1) ** 2);
  let best = -2, bi = 0, bj = 0;
  for (let dj = 0; dj <= 2 * S; dj++) for (let di = 0; di <= 2 * S; di++) {
    let sm = 0, s2 = 0, cr = 0;
    for (let j = 0, k = 0; j < n; j++) {
      const row = (dj + j) * span + di;
      for (let i = 0; i < n; i++, k++) { const v = Wd[row + i]; sm += v; s2 += v * v; cr += v * P[k]; }
    }
    const vv = s2 - (sm * sm) / N;
    const c = vv > 1e-9 ? cr / Math.sqrt(vv * pv) : -1;
    sc[dj * (2 * S + 1) + di] = c;
    if (c > best) { best = c; bi = di; bj = dj; }
  }
  if (bi === 0 || bj === 0 || bi === 2 * S || bj === 2 * S) return null;  // on the window edge
  const at = (i, j) => sc[j * (2 * S + 1) + i];
  const sub = (a, b, c) => { const d = a - 2 * b + c; return d < 0 ? Math.max(-0.5, Math.min(0.5, 0.5 * (a - c) / d)) : 0; };
  const dx = bi - S + sub(at(bi - 1, bj), best, at(bi + 1, bj));
  const dy = bj - S + sub(at(bi, bj - 1), best, at(bi, bj + 1));
  return { dx, dy, score: best };
}

// Pick well-spread reference points (fine px) for refinement: corners from the
// reference features, inside the ROI, at most `max`, on a grid.
// only: optional list of work-px reference points to choose from (the matches on
// the chosen surface), so refinement can't wander onto another surface.
function refinePoints(ref, roi, near, only = null, max = 260) {
  const f = ref.feats, k = ref.toFine / ref.toWork, pts = [];
  const lim = near ? 0.45 * Math.max(ref.W, ref.H) : Infinity;
  const cell = Math.max(ref.fine.w, ref.fine.h) / (only ? 30 : 18), used = new Set();
  const cand = only ? only.map(([x, y]) => [x * k, y * k]) :
    [...Array(f.n).keys()].sort((a, b) => f.scale[a] - f.scale[b]).map(i => [f.x[i] * k, f.y[i] * k]);
  for (const [x, y] of cand) {
    if (roi && !inRoi(roi, x / ref.toFine, y / ref.toFine)) continue;
    if (near && Math.hypot(x / ref.toFine - near[0], y / ref.toFine - near[1]) > lim) continue;
    const key = ((y / cell) | 0) * 1000 + ((x / cell) | 0);
    if (used.has(key)) continue;
    used.add(key); pts.push([x, y]);
    if (pts.length >= max) break;
  }
  return pts;
}

const inRoi = (roi, x, y) => x >= roi.x && y >= roi.y && x <= roi.x + roi.w && y <= roi.y + roi.h;

// G: ref fine -> tgt fine. Returns refined G or null.
function refine(ref, tgt, G, kind, roi, near = null, only = null) {
  let pts = refinePoints(ref, roi, near, only);
  if (pts.length < 30 && near) pts = refinePoints(ref, roi, null, only);
  if (pts.length < 12 && only) pts = refinePoints(ref, roi, near);
  if (pts.length < 6) return null;
  let cur = G, inl = 0, rms = 0;
  for (const [S, r] of [[6, 9], [2, 7]]) {
    const src = [], dst = [], wts = [];
    for (const [x, y] of pts) {
      const m = nccShift(ref.fine, tgt.fine, cur, x, y, r, S);
      if (!m || m.score < 0.6) continue;
      const [u, v] = apply(cur, x + m.dx, y + m.dy);
      src.push(x, y); dst.push(u, v); wts.push(m.score);
    }
    if (src.length / 2 < 8) return null;
    const res = ransac(kind, src, dst, { thresh: 1.5, iters: 400, weights: wts, w: ref.fine.w, h: ref.fine.h });
    if (!res) return null;
    cur = res.H; inl = res.inliers.length;
    const e = residuals(res.H, src, dst, res.inliers);
    rms = Math.sqrt(e.reduce((a, b) => a + b * b, 0) / e.length);
  }
  return { G: cur, inliers: inl, rms };
}

// ---------------------------------------------------------------- pairwise

// conv(a, b): full px of image a <- full px scaled by k
const S = k => scaling(k);

// Align target to reference. kind: 'translate' | 'similarity' | 'homography'.
// A full perspective map is always estimated (photos taken from slightly different
// spots are perspective views of each other), weighted towards the anchor so it
// locks onto the surface the anchor sits on; 'translate' and 'similarity' then keep
// only the shift / turn+scale of that map at the anchor, so the anchor itself lands
// exactly in every mode.
// roi (ref full px) limits matching to the subject.
// Returns { H (ref full -> tgt full), inliers, matches, rms (ref px), ok } or { ok: false }.
export function alignPair(ref, tgt, { kind = 'homography', roi = null, anchor = null } = {}) {
  const A = ref.feats, B = tgt.feats;
  const mt = matchFeatures(A, B);
  const src = [], dst = [], wts = [];
  const a = anchor || [ref.W / 2, ref.H / 2];
  const sig = 0.35 * Math.max(ref.W, ref.H);
  for (let k = 0; k < mt.a.length; k++) {
    const i = mt.a[k], j = mt.b[k];
    const x = A.x[i] / ref.toWork, y = A.y[i] / ref.toWork;
    if (roi && !inRoi(roi, x, y)) continue;
    src.push(A.x[i], A.y[i]); dst.push(B.x[j], B.y[j]);
    wts.push(roi ? 1 : 0.15 + Math.exp(-((x - a[0]) ** 2 + (y - a[1]) ** 2) / (2 * sig * sig)));
  }
  const n = src.length / 2;
  if (n < 8) return { ok: false, matches: n, reason: 'too few matches' };
  const thresh = Math.max(2.5, Math.max(ref.work.w, ref.work.h) * 0.004);
  let est = 'homography';
  let rs = roi ? ransac(est, src, dst, { thresh, iters: 3000, w: ref.work.w, h: ref.work.h })
    : surfaceAtAnchor(src, dst, wts, [a[0] * ref.toWork, a[1] * ref.toWork], thresh, ref.work, tgt.work);
  if (!rs || rs.inliers.length < 12) {
    // too little for perspective: a similarity still lines most sets up
    est = 'similarity';
    rs = ransac(est, src, dst, { thresh: thresh * 1.5, iters: 2000, weights: wts, w: ref.work.w, h: ref.work.h });
  }
  if (!rs || rs.inliers.length < 8) {
    return { ok: false, matches: n, inliers: rs ? rs.inliers.length : 0, reason: 'no consistent match' };
  }
  // work -> fine maps, then sub-pixel refinement near the subject
  const Rf = ref.toFine / ref.toWork, Tf = tgt.toFine / tgt.toWork;
  let G = mul3(S(Tf), mul3(rs.H, S(1 / Rf)));
  const onSurface = rs.inliers.map(i => [src[2 * i], src[2 * i + 1]]);
  const rf = refine(ref, tgt, G, est, roi, roi ? null : a, onSurface);
  let rms = null, inl = rs.inliers.length;
  if (rf && rf.inliers >= 8) { G = rf.G; rms = rf.rms / ref.toFine; inl = rf.inliers; }
  // fine -> full
  let H = mul3(S(1 / tgt.toFine), mul3(G, S(ref.toFine)));
  H = reduceTo(kind, H, a);
  return { ok: true, H, inliers: inl, matches: n, rms, refined: !!rf, model: est };
}

// Scenes with depth (a building corner seen from two spots) hold several surfaces,
// each with its own perspective map. Peel them off one by one and keep the one whose
// matches crowd closest round the anchor; flat scenes just return the first.
function surfaceAtAnchor(src, dst, wts, [ax, ay], thresh, work, workT) {
  let idx = [...Array(src.length / 2).keys()];
  const cands = [];
  for (let k = 0; k < 3 && idx.length >= 16; k++) {
    const s2 = [], d2 = [], w2 = [];
    for (const i of idx) { s2.push(src[2 * i], src[2 * i + 1]); d2.push(dst[2 * i], dst[2 * i + 1]); w2.push(wts[i]); }
    const rs = ransac('homography', s2, d2, { thresh, iters: 3000, weights: w2, w: work.w, h: work.h });
    if (!rs || rs.inliers.length < 12) break;
    cands.push({ H: rs.H, n: rs.inliers.length });
    const used = new Set(rs.inliers);
    idx = idx.filter((_, j) => !used.has(j));
  }
  if (!cands.length) return null;
  // the surface whose map makes the patch round the anchor match best
  let best = cands[0];
  if (cands.length > 1) {
    const r = Math.max(12, Math.round(0.03 * Math.max(work.w, work.h)));
    let bestS = -2;
    for (const c of cands) {
      const sc = patchNCC(work, workT, c.H, ax, ay, r);
      c.score = sc;
      if (sc > bestS + 0.02) { bestS = sc; best = c; }
    }
  }
  // every match consistent with the chosen surface
  const t2 = thresh * thresh, inl = [];
  for (let i = 0; i < src.length / 2; i++) {
    const [u, v] = apply(best.H, src[2 * i], src[2 * i + 1]);
    if ((u - dst[2 * i]) ** 2 + (v - dst[2 * i + 1]) ** 2 < t2) inl.push(i);
  }
  return { H: best.H, inliers: inl };
}

// NCC between the reference patch round (x, y) and the target sampled through H.
function patchNCC(R, T, H, x, y, r) {
  let n = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  for (let j = -r; j <= r; j += 2) for (let i = -r; i <= r; i += 2) {
    const a = sample(R, x + i, y + j);
    const [u, v] = apply(H, x + i, y + j);
    const b = sample(T, u, v);
    if (a !== a || b !== b) continue;
    n++; sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
  }
  if (n < 20) return -1;
  const va = saa - sa * sa / n, vb = sbb - sb * sb / n;
  return va > 1e-9 && vb > 1e-9 ? (sab - sa * sb / n) / Math.sqrt(va * vb) : -1;
}

export function reduceTo(kind, H, anchor) {
  if (kind === 'translate') return reduceToTranslation(H, anchor);
  if (kind === 'similarity') return reduceToSimilarity(H, anchor);
  return H;
}

// Turn + uniform scale of H at the anchor (closest rotation-scale to its Jacobian),
// anchor mapped exactly.
export function reduceToSimilarity(H, [ax, ay]) {
  const w = H[6] * ax + H[7] * ay + H[8];
  const [u, v] = apply(H, ax, ay);
  const j00 = (H[0] - H[6] * u) / w, j01 = (H[1] - H[7] * u) / w;
  const j10 = (H[3] - H[6] * v) / w, j11 = (H[4] - H[7] * v) / w;
  const c = (j00 + j11) / 2, s = (j10 - j01) / 2;   // nearest [[c,-s],[s,c]]
  return [c, -s, u - c * ax + s * ay, s, c, v - s * ax - c * ay, 0, 0, 1];
}

// Keep only the shift that puts the anchor where H puts it (sun-style centring).
export function reduceToTranslation(H, [ax, ay]) {
  const [u, v] = apply(H, ax, ay);
  return translation(u - ax, v - ay);
}

// ---------------------------------------------------------------- brightest spot
// The sun / a lamp: centroid of the saturated blob around the brightest point.
// lum: gray at any scale; returns full-res [x, y] or null.
export function brightestSpot(lum, W, H, near = null, radius = null) {
  const k = Math.min(1, 640 / Math.max(lum.w, lum.h));
  const im = k < 1 ? downscale(lum, Math.round(lum.w * k), Math.round(lum.h * k)) : lum;
  const sm = blur(im, Math.max(1, im.w / 300));
  const { data, w, h } = sm, sx = w / W;
  let best = -1, bi = -1;
  for (let i = 0; i < data.length; i++) {
    if (near && radius) {
      const x = (i % w) / sx, y = ((i / w) | 0) / sx;
      if ((x - near[0]) ** 2 + (y - near[1]) ** 2 > radius * radius) continue;
    }
    if (data[i] > best) { best = data[i]; bi = i; }
  }
  if (bi < 0) return null;
  const t = best - Math.max(0.02, best * 0.04);
  // flood fill the blob above t from the peak
  const seen = new Uint8Array(w * h), stack = [bi];
  seen[bi] = 1;
  let mx = 0, my = 0, ms = 0, count = 0;
  while (stack.length) {
    const i = stack.pop(), x = i % w, y = (i / w) | 0, wgt = data[i] - t + 1e-3;
    mx += x * wgt; my += y * wgt; ms += wgt; count++;
    if (count > w * h * 0.15) break;     // not a spot: half the sky is white
    for (const j of [i - 1, i + 1, i - w, i + w]) {
      if (j < 0 || j >= w * h || seen[j]) continue;
      if ((j % w === 0 && i % w === w - 1) || (i % w === 0 && j % w === w - 1)) continue;
      if (data[j] >= t) { seen[j] = 1; stack.push(j); }
    }
  }
  return { x: (mx / ms + 0.5) / sx - 0.5 / sx, y: (my / ms + 0.5) / sx - 0.5 / sx, peak: best, area: count / (sx * sx) };
}

// ---------------------------------------------------------------- the whole stack

// items: [{ prep, lum, W, H, manual?: [x,y] (anchor in this image, full px) }]
// opts: { ref, method: 'match'|'bright'|'manual', kind, roi, anchor, onProgress }
// Returns per item { H, ok, status, inliers, rms, via }.
export async function alignStack(items, opts) {
  const { ref = 0, method = 'match', kind = 'homography', roi = null, onProgress = null } = opts;
  const R = items[ref];
  const anchor = opts.anchor || [R.W / 2, R.H / 2];
  const out = items.map(() => null);
  out[ref] = { H: I3(), ok: true, status: 'reference', inliers: null, rms: 0 };
  const total = items.length;
  let done = 1;
  const tick = async label => { onProgress?.({ p: done / total, label }); await null; };

  if (method === 'bright') {
    const a = brightestSpot(R.lum, R.W, R.H);
    for (let i = 0; i < items.length; i++) {
      if (i === ref) continue;
      const it = items[i];
      const b = brightestSpot(it.lum, it.W, it.H);
      const sx = it.W / R.W;  // same framing at a different resolution
      out[i] = b && a
        ? { H: mul3(translation(b.x - a.x * sx, b.y - a.y * sx), scaling(sx)), ok: true, status: 'spot', spot: [b.x, b.y] }
        : { H: scaling(sx), ok: false, status: 'no bright spot' };
      done++; await tick(`spot ${done}/${total}`);
    }
    out[ref].spot = a ? [a.x, a.y] : null;
    return { results: out, anchor: a ? [a.x, a.y] : anchor };
  }

  if (method === 'manual') {
    const a = R.manual || anchor;
    for (let i = 0; i < items.length; i++) {
      if (i === ref) continue;
      const it = items[i], sx = it.W / R.W;
      out[i] = it.manual
        ? { H: mul3(translation(it.manual[0] - a[0] * sx, it.manual[1] - a[1] * sx), scaling(sx)), ok: true, status: 'by hand' }
        : { H: scaling(sx), ok: false, status: 'needs a click' };
    }
    return { results: out, anchor: a };
  }

  // feature matching: direct to the reference first
  const failed = [];
  for (let i = 0; i < items.length; i++) {
    if (i === ref) continue;
    const r = alignPair(R.prep, items[i].prep, { kind, roi, anchor });
    out[i] = r.ok ? { ...r, status: 'matched', via: null } : { ...r, status: r.reason };
    if (!r.ok) failed.push(i);
    done++; await tick(`matched ${done}/${total}`);
  }
  // chain the failures through any image that did match: ref -> j -> i
  for (const i of failed) {
    let best = null;
    for (let j = 0; j < items.length; j++) {
      if (j === i || j === ref || !out[j]?.ok || out[j].via != null) continue;
      // ROI carried into image j as the bbox of its mapped quad
      let roiJ = null;
      if (roi) {
        const q = [[roi.x, roi.y], [roi.x + roi.w, roi.y], [roi.x + roi.w, roi.y + roi.h], [roi.x, roi.y + roi.h]].map(([x, y]) => apply(out[j].H, x, y));
        const xs = q.map(p => p[0]), ys = q.map(p => p[1]);
        roiJ = { x: Math.min(...xs), y: Math.min(...ys), w: Math.max(...xs) - Math.min(...xs), h: Math.max(...ys) - Math.min(...ys) };
      }
      const r = alignPair(items[j].prep, items[i].prep, { kind: 'homography', roi: roiJ, anchor: apply(out[j].H, ...anchor) });
      if (r.ok && (!best || r.inliers > best.r.inliers)) best = { r, j };
    }
    if (best) {
      const H = reduceTo(kind, mul3(best.r.H, out[best.j].H), anchor);
      out[i] = { ...best.r, H, status: 'matched via ' + (best.j + 1), via: best.j };
    }
  }
  for (let i = 0; i < out.length; i++) {
    if (!out[i].H) out[i].H = scaling(items[i].W / R.W);
    if (out[i].ok && !sane(inv3(out[i].H), items[i].W, items[i].H)) { out[i].ok = false; out[i].status = 'implausible fit'; }
  }
  return { results: out, anchor };
}

// Move image i's map so the anchor lands exactly on a hand-placed point in it
// (a nudge on top of any automatic fit).
export function pinAnchor(H, anchor, target) {
  const [u, v] = apply(H, anchor[0], anchor[1]);
  return mul3(translation(target[0] - u, target[1] - v), H);
}

// ---------------------------------------------------------------- crops

export const ASPECTS = [
  { id: '16:9', r: 16 / 9 }, { id: '9:16', r: 9 / 16 }, { id: '1:1', r: 1 },
  { id: '4:5', r: 4 / 5 }, { id: '5:4', r: 5 / 4 }, { id: '3:2', r: 3 / 2 }, { id: '2:3', r: 2 / 3 },
  { id: '4:3', r: 4 / 3 }, { id: '3:4', r: 3 / 4 }, { id: '21:9', r: 21 / 9 },
];

// Footprint of image i in the reference frame.
export const footprint = (H, w, h) => {
  const Hi = inv3(H);
  return Hi ? quadOf(Hi, w, h) : null;
};

// Region every included image covers (reference px).
export function commonRegion(items, maps, include) {
  const polys = [];
  items.forEach((it, i) => { if (include[i] && maps[i]) { const q = footprint(maps[i], it.W, it.H); if (q) polys.push(q); } });
  return intersectAll(polys);
}

// Native output size: the largest output that doesn't upscale any included image.
export function nativeScale(items, maps, include, rect) {
  const cx = rect.x + rect.w / 2, cy = rect.y + rect.h / 2;
  let s = Infinity;
  items.forEach((it, i) => {
    if (!include[i] || !maps[i]) return;
    // sample the rect's corners + centre: worst (smallest) local scale wins
    for (const [x, y] of [[cx, cy], [rect.x, rect.y], [rect.x + rect.w, rect.y], [rect.x, rect.y + rect.h], [rect.x + rect.w, rect.y + rect.h]]) {
      s = Math.min(s, localScale(maps[i], x, y));
    }
  });
  return isFinite(s) ? s : 1;
}

// bestRect plus 'thirds': the rule-of-thirds point that gives the biggest frame.
export function placeRect(V, r, anchor, place) {
  if (place !== 'thirds') return bestRect(V, r, anchor, place);
  let best = null;
  for (const f of [[1 / 3, 1 / 3], [2 / 3, 1 / 3], [1 / 3, 2 / 3], [2 / 3, 2 / 3]]) {
    const q = bestRect(V, r, anchor, f);
    if (q && (!best || q.w > best.w)) best = q;
  }
  return best;
}

// Everything the crop panel shows: per aspect the best rect, its size in reference
// px and native px, and per image how much bigger the crop gets without it.
export function cropReport(items, maps, include, anchor, { place = 'centre', zoom = 1, aspects = ASPECTS } = {}) {
  const V = commonRegion(items, maps, include);
  const rows = aspects.map(a => {
    const rect = zoomRect(placeRect(V, a.r, anchor, place), anchor, zoom);
    if (!rect) return { ...a, rect: null };
    const k = nativeScale(items, maps, include, rect);
    return { ...a, rect, refW: rect.w, refH: rect.h, outW: Math.floor(rect.w * k), outH: Math.floor(rect.h * k) };
  });
  return { region: V, area: V.length ? Math.abs(polyArea(V)) : 0, rows };
}

// For the chosen aspect: area gain if each included image were left out.
export function leaveOneOut(items, maps, include, anchor, r, place = 'centre') {
  const base = placeRect(commonRegion(items, maps, include), r, anchor, place);
  const baseA = base ? base.w * base.h : 0;
  return items.map((_, i) => {
    if (!include[i]) return null;
    const inc = include.slice(); inc[i] = false;
    if (inc.filter(Boolean).length < 1) return null;
    const rr = placeRect(commonRegion(items, maps, inc), r, anchor, place);
    return rr && baseA ? (rr.w * rr.h) / baseA : null;
  });
}

export { apply, inv3, mul3, I3, fitSimilarity, fitModel };
