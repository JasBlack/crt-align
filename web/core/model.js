// Warp model: canvas (original image) -> filmed photo.
//
//   canvas (u, v) in [0,1]^2
//     -> content warp   s = u + eu(u) * (A_s(u,v) - u + Bx(u,v))
//                       t = v + ev(v) * (A_t(u,v) - v + By(u,v))
//        A = residual affine (overscan zoom / offset / tilt),
//        B = cubic B-spline mesh (the "mesh warp"),
//        eu/ev = edge envelopes: 0 on the frame edge, so corners stay pinned and
//        edge content can only slide along the edge, never off it.
//     -> inset          s' = inset + s (1 - 2 inset)        (crop into the screen)
//     -> screen         homography of the 4 screen corners + parabolic bulge of each
//                       edge (curved CRT glass)
//   = filmed pixel (X, Y) in full-resolution filmed coordinates.

export const AFFINE_N = 6;

export function smoothstep01(t) {
  if (t >= 1) return 1;
  if (t <= 0) return 0;
  return t * t * (3 - 2 * t);
}

export function envelope(u, m) {
  if (m <= 0) return 1;
  return smoothstep01(u / m) * smoothstep01((1 - u) / m);
}

export function bsplineWeights(f, out, o = 0) {
  const f2 = f * f, f3 = f2 * f, g = 1 - f;
  out[o] = (g * g * g) / 6;
  out[o + 1] = (3 * f3 - 6 * f2 + 4) / 6;
  out[o + 2] = (-3 * f3 + 3 * f2 + 3 * f + 1) / 6;
  out[o + 3] = f3 / 6;
}

// Knot segment + fraction for a cubic uniform B-spline with n intervals on [0,1].
export function bsplineIndex(u, n) {
  let s = u * n;
  let k = Math.floor(s);
  if (k < 0) k = 0;
  if (k > n - 1) k = n - 1;
  return [k, s - k];
}

export function gridForAspect(nLong, w, h) {
  if (w >= h) return { nx: nLong, ny: Math.max(2, Math.round(nLong * h / w)) };
  return { nx: Math.max(2, Math.round(nLong * w / h)), ny: nLong };
}

// ---------------------------------------------------------------- screen geometry

// Heckbert square-to-quad homography: (0,0),(1,0),(1,1),(0,1) -> TL, TR, BR, BL.
export function squareToQuad(c) {
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = c;
  const dx1 = x1 - x2, dx2 = x3 - x2, dx3 = x0 - x1 + x2 - x3;
  const dy1 = y1 - y2, dy2 = y3 - y2, dy3 = y0 - y1 + y2 - y3;
  let a, b, d, e, g, h;
  if (Math.abs(dx3) < 1e-9 && Math.abs(dy3) < 1e-9) {
    g = 0; h = 0;
    a = x1 - x0; b = x2 - x1; d = y1 - y0; e = y2 - y1;
  } else {
    const den = dx1 * dy2 - dy1 * dx2;
    g = (dx3 * dy2 - dy3 * dx2) / den;
    h = (dx1 * dy3 - dy1 * dx3) / den;
    a = x1 - x0 + g * x1; b = x3 - x0 + h * x3;
    d = y1 - y0 + g * y1; e = y3 - y0 + h * y3;
  }
  return [a, b, x0, d, e, y0, g, h, 1];
}

// Outward unit normals of the 4 edges [top, right, bottom, left].
export function edgeNormals(c) {
  const cx = (c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4;
  const cy = (c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4;
  const pairs = [[0, 1], [1, 2], [3, 2], [0, 3]];
  return pairs.map(([i, j]) => {
    const dx = c[j][0] - c[i][0], dy = c[j][1] - c[i][1];
    const len = Math.hypot(dx, dy) || 1;
    let nx = dy / len, ny = -dx / len;
    const mx = (c[i][0] + c[j][0]) / 2 - cx, my = (c[i][1] + c[j][1]) / 2 - cy;
    if (nx * mx + ny * my < 0) { nx = -nx; ny = -ny; }
    return [nx, ny];
  });
}

// Precomputed screen map from {corners, bulge}.
export function makeScreen(corners, bulge) {
  return { corners, bulge: bulge.slice(), H: squareToQuad(corners), N: edgeNormals(corners) };
}

// Screen point + partial derivatives. out = [X, Y, Xs, Ys, Xt, Yt].
export function screenEval(S, s, t, out) {
  const [a, b, c, d, e, f, g, h] = S.H;
  const D = g * s + h * t + 1;
  const N1 = a * s + b * t + c, N2 = d * s + e * t + f;
  const iD = 1 / D, iD2 = iD * iD;
  let X = N1 * iD, Y = N2 * iD;
  let Xs = (a * D - N1 * g) * iD2, Ys = (d * D - N2 * g) * iD2;
  let Xt = (b * D - N1 * h) * iD2, Yt = (e * D - N2 * h) * iD2;
  const B = S.bulge, N = S.N;
  const ss = 4 * s * (1 - s), tt = 4 * t * (1 - t);
  const dss = 4 * (1 - 2 * s), dtt = 4 * (1 - 2 * t);
  // weights and their s/t partials for [top, right, bottom, left]
  const w0 = ss * (1 - t), w0s = dss * (1 - t), w0t = -ss;
  const w1 = tt * s, w1s = tt, w1t = dtt * s;
  const w2 = ss * t, w2s = dss * t, w2t = ss;
  const w3 = tt * (1 - s), w3s = -tt, w3t = dtt * (1 - s);
  const k0 = B[0], k1 = B[1], k2 = B[2], k3 = B[3];
  X += w0 * k0 * N[0][0] + w1 * k1 * N[1][0] + w2 * k2 * N[2][0] + w3 * k3 * N[3][0];
  Y += w0 * k0 * N[0][1] + w1 * k1 * N[1][1] + w2 * k2 * N[2][1] + w3 * k3 * N[3][1];
  Xs += w0s * k0 * N[0][0] + w1s * k1 * N[1][0] + w2s * k2 * N[2][0] + w3s * k3 * N[3][0];
  Ys += w0s * k0 * N[0][1] + w1s * k1 * N[1][1] + w2s * k2 * N[2][1] + w3s * k3 * N[3][1];
  Xt += w0t * k0 * N[0][0] + w1t * k1 * N[1][0] + w2t * k2 * N[2][0] + w3t * k3 * N[3][0];
  Yt += w0t * k0 * N[0][1] + w1t * k1 * N[1][1] + w2t * k2 * N[2][1] + w3t * k3 * N[3][1];
  out[0] = X; out[1] = Y; out[2] = Xs; out[3] = Ys; out[4] = Xt; out[5] = Yt;
  return out;
}

// ---------------------------------------------------------------- model

export function createModel({ W, H, FW, FH, corners, bulge, inset = 0.02, margin = 0.05,
                              nx = 2, ny = 2 }) {
  const m = {
    W, H, FW, FH,
    screen: makeScreen(corners, bulge),
    inset,
    // edge band width per axis [u, v]: where the content warp fades to the pinned frame
    margin: Array.isArray(margin) ? margin.slice() : [margin, margin],
    affine: new Float64Array([1, 0, 0, 0, 1, 0]),
    nx, ny,
    cx: new Float64Array((nx + 3) * (ny + 3)),
    cy: new Float64Array((nx + 3) * (ny + 3)),
    pins: [],
  };
  return m;
}

export function setScreen(model, corners, bulge) {
  model.screen = makeScreen(corners, bulge);
}

export function cloneModel(m) {
  return {
    ...m,
    screen: makeScreen(m.screen.corners.map(p => p.slice()), m.screen.bulge),
    affine: m.affine.slice(),
    margin: m.margin.slice(),
    cx: m.cx.slice(), cy: m.cy.slice(),
    pins: m.pins.map(p => ({ ...p })),
  };
}

export function paramCount(m) {
  return AFFINE_N + 2 * m.cx.length;
}

export function getParams(m) {
  const p = new Float64Array(paramCount(m));
  p.set(m.affine, 0);
  p.set(m.cx, AFFINE_N);
  p.set(m.cy, AFFINE_N + m.cx.length);
  return p;
}

export function setParams(m, p) {
  m.affine.set(p.subarray(0, AFFINE_N));
  m.cx.set(p.subarray(AFFINE_N, AFFINE_N + m.cx.length));
  m.cy.set(p.subarray(AFFINE_N + m.cx.length));
}

const _sc = new Float64Array(6), _wu = new Float64Array(4), _wv = new Float64Array(4);

// Map one canvas point. Returns [X, Y] in full-res filmed pixels.
export function mapPoint(m, u, v, out = [0, 0]) {
  const nx = m.nx, stride = nx + 3;
  const [ku, fu] = bsplineIndex(u, nx), [kv, fv] = bsplineIndex(v, m.ny);
  bsplineWeights(fu, _wu); bsplineWeights(fv, _wv);
  let bx = 0, by = 0;
  for (let b = 0; b < 4; b++) {
    const row = (kv + b) * stride + ku;
    for (let a = 0; a < 4; a++) {
      const w = _wu[a] * _wv[b];
      bx += w * m.cx[row + a]; by += w * m.cy[row + a];
    }
  }
  const A = m.affine;
  const du = u - 0.5, dv = v - 0.5;
  const sA = 0.5 + A[0] * du + A[1] * dv + A[2];
  const tA = 0.5 + A[3] * du + A[4] * dv + A[5];
  const s = u + envelope(u, m.margin[0]) * (sA - u + bx);
  const t = v + envelope(v, m.margin[1]) * (tA - v + by);
  const k = 1 - 2 * m.inset;
  screenEval(m.screen, m.inset + s * k, m.inset + t * k, _sc);
  out[0] = _sc[0]; out[1] = _sc[1];
  return out;
}

// Plain JSON for saving / passing to the GL renderer.
export function modelToJSON(m) {
  return {
    version: 1, W: m.W, H: m.H, FW: m.FW, FH: m.FH,
    corners: m.screen.corners, bulge: m.screen.bulge,
    inset: m.inset, margin: m.margin.slice(),
    affine: Array.from(m.affine), nx: m.nx, ny: m.ny,
    cx: Array.from(m.cx), cy: Array.from(m.cy),
    pins: m.pins,
  };
}

export function modelFromJSON(j) {
  const m = createModel({ W: j.W, H: j.H, FW: j.FW, FH: j.FH, corners: j.corners,
                          bulge: j.bulge, inset: j.inset, margin: j.margin,
                          nx: j.nx, ny: j.ny });
  m.affine.set(j.affine); m.cx.set(j.cx); m.cy.set(j.cy);
  m.pins = (j.pins || []).map(p => ({ ...p }));
  return m;
}

// Least-squares refit of the B-spline mesh onto a new grid size.
export function refitGrid(m, nx2, ny2) {
  if (nx2 === m.nx && ny2 === m.ny) return;
  const s2 = nx2 + 3, nc = s2 * (ny2 + 3);
  const ATA = new Float64Array(nc * nc), ATbx = new Float64Array(nc), ATby = new Float64Array(nc);
  const S = 6;
  const wu = new Float64Array(4), wv = new Float64Array(4);
  const ow = new Float64Array(4), ov = new Float64Array(4);
  const idx = new Int32Array(16), wts = new Float64Array(16);
  const sx = nx2 * S, sy = ny2 * S;
  for (let j = 0; j <= sy; j++) {
    const v = j / sy;
    for (let i = 0; i <= sx; i++) {
      const u = i / sx;
      // old field value
      const [ku0, fu0] = bsplineIndex(u, m.nx), [kv0, fv0] = bsplineIndex(v, m.ny);
      bsplineWeights(fu0, ow); bsplineWeights(fv0, ov);
      let bx = 0, by = 0;
      for (let b = 0; b < 4; b++) for (let a = 0; a < 4; a++) {
        const c = (kv0 + b) * (m.nx + 3) + ku0 + a, w = ow[a] * ov[b];
        bx += w * m.cx[c]; by += w * m.cy[c];
      }
      const [ku, fu] = bsplineIndex(u, nx2), [kv, fv] = bsplineIndex(v, ny2);
      bsplineWeights(fu, wu); bsplineWeights(fv, wv);
      let n = 0;
      for (let b = 0; b < 4; b++) for (let a = 0; a < 4; a++) {
        idx[n] = (kv + b) * s2 + ku + a; wts[n] = wu[a] * wv[b]; n++;
      }
      for (let p = 0; p < 16; p++) {
        const ip = idx[p], wp = wts[p];
        ATbx[ip] += wp * bx; ATby[ip] += wp * by;
        for (let q = 0; q < 16; q++) ATA[ip * nc + idx[q]] += wp * wts[q];
      }
    }
  }
  for (let i = 0; i < nc; i++) ATA[i * nc + i] += 1e-6;
  const L = cholesky(ATA, nc);
  m.nx = nx2; m.ny = ny2;
  m.cx = cholSolve(L, nc, ATbx);
  m.cy = cholSolve(L, nc, ATby);
}

function cholesky(A, n) {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      L[i * n + j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j * n + j];
    }
  }
  return L;
}

function cholSolve(L, n, b) {
  const y = new Float64Array(n), x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k];
    y[i] = s / L[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}

// Content warp (u,v) -> screen params (s,t) WITHOUT the edge band: where the original's
// content actually is on the screen. The optimizer fits this; the band is applied only
// when rendering (mapPoint), so pinning the frame never distorts the matched interior.
export function contentST(m, u, v, out = [0, 0]) {
  const nx = m.nx, stride = nx + 3;
  const [ku, fu] = bsplineIndex(u, nx), [kv, fv] = bsplineIndex(v, m.ny);
  bsplineWeights(fu, _wu); bsplineWeights(fv, _wv);
  let bx = 0, by = 0;
  for (let b = 0; b < 4; b++) for (let a = 0; a < 4; a++) {
    const c = (kv + b) * stride + ku + a, w = _wu[a] * _wv[b];
    bx += w * m.cx[c]; by += w * m.cy[c];
  }
  const A = m.affine, du = u - 0.5, dv = v - 0.5;
  out[0] = 0.5 + A[0] * du + A[1] * dv + A[2] + bx;
  out[1] = 0.5 + A[3] * du + A[4] * dv + A[5] + by;
  return out;
}

// Render warp (u,v) -> (s,t) with the edge band: identity on the frame edge.
export function renderST(m, u, v, out = [0, 0]) {
  contentST(m, u, v, out);
  out[0] = u + envelope(u, m.margin[0]) * (out[0] - u);
  out[1] = v + envelope(v, m.margin[1]) * (out[1] - v);
  return out;
}

// Edge bands just wide enough for a smooth, fold-free hand-off from the matched
// content to the pinned frame. Overscan pushes the content's edge beyond the screen;
// the band has to absorb that overshoot (~1.8x for the smoothstep ramp).
export function safeMargins(m, minMargin = 0.03) {
  let du = 0, dv = 0;
  const p = [0, 0];
  for (let k = 0; k <= 40; k++) {
    const q = k / 40;
    contentST(m, 0, q, p); du = Math.max(du, Math.abs(p[0]));
    contentST(m, 1, q, p); du = Math.max(du, Math.abs(p[0] - 1));
    contentST(m, q, 0, p); dv = Math.max(dv, Math.abs(p[1]));
    contentST(m, q, 1, p); dv = Math.max(dv, Math.abs(p[1] - 1));
  }
  const f = d => Math.min(0.3, Math.max(minMargin, 1.8 * d + 0.015));
  return [f(du), f(dv)];
}

// Minimum Jacobian determinant of the render warp (u,v) -> (s,t). <= 0: it folds.
export function minFoldDet(m, n = 64) {
  const h = 1 / n, p = [0, 0], pu = [0, 0], pv = [0, 0];
  let worst = Infinity;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const u = (i + 0.5) * h, v = (j + 0.5) * h, e = 0.25 * h;
    renderST(m, u, v, p); renderST(m, u + e, v, pu); renderST(m, u, v + e, pv);
    const det = ((pu[0] - p[0]) * (pv[1] - p[1]) - (pv[0] - p[0]) * (pu[1] - p[1])) / (e * e);
    worst = Math.min(worst, det);
  }
  return worst;
}

// Choose edge bands for rendering: smallest safe ones, widened until fold-free.
export function fitMargins(m) {
  m.margin = safeMargins(m);
  for (let k = 0; k < 8 && minFoldDet(m) < 0.05; k++) {
    m.margin = m.margin.map(x => Math.min(0.4, x * 1.25));
  }
  return m.margin;
}
