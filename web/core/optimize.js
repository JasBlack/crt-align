// Alignment optimizer.
//
// Similarity: normalized gradient fields (NGF). For every canvas pixel the unit-ish
// edge direction of the original is compared with the edge direction of the warped
// filmed image; score = mean (nO . nW)^2. Squared dot -> blind to contrast inversion,
// false colour, emboss and brightness changes; only edge *geometry* matters. Flat
// regions carry ~no weight, so glitch noise there cannot pull the mesh around.
//
// Parameters: residual affine (6) + B-spline mesh (2 x (nx+3)(ny+3)), optimized
// coarse-to-fine with L-BFGS and analytic gradients through the whole warp chain.

import { resize, downscale, blur, gradient, pyramid, gradientScale } from './image.js';
import {
  AFFINE_N, envelope, bsplineWeights, bsplineIndex, screenEval, makeScreen,
  createModel, cloneModel, getParams, setParams, gridForAspect, refitGrid, fitMargins,
} from './model.js';
import { detectScreen } from './screen.js';

export const QUALITY = {
  quick:  [{ w: 128, n: 3, it: 20 }],
  coarse: [{ w: 128, n: 3, it: 30 }, { w: 256, n: 5, it: 30 }],
  medium: [{ w: 128, n: 3, it: 30 }, { w: 256, n: 6, it: 30 }, { w: 512, n: 10, it: 30 }],
  fine:   [{ w: 128, n: 3, it: 30 }, { w: 256, n: 6, it: 30 }, { w: 512, n: 10, it: 30 },
           { w: 768, n: 16, it: 20 }],
};

export const DEFAULTS = { lambda: 0.3, pinStrength: 0.05, margin: 'auto' };

// Yield to the UI between iterations on the main thread. In a worker (or Node)
// there is nothing to yield to, and background-tab timer throttling would only
// slow the solve down.
const tick = typeof window === 'undefined' ? () => Promise.resolve() : () => new Promise(r => setTimeout(r, 0));

// ---------------------------------------------------------------- level data

function filmLevelFor(ctx, ratio) {
  // pick the finest pyramid level whose pixels are not finer than the canvas sampling
  const { filmPyr, FW } = ctx;
  let best = 0;
  for (let l = 0; l < filmPyr.length; l++) {
    if (FW / filmPyr[l].w <= ratio * 1.05) best = l;
  }
  return best;
}

function screenWidthFull(m) {
  const c = m.screen.corners;
  return (Math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1]) +
          Math.hypot(c[2][0] - c[3][0], c[2][1] - c[3][1])) / 2;
}

export function buildLevel(ctx, m, w) {
  const { orig } = ctx;
  w = Math.min(w, orig.w);
  const h = Math.max(8, Math.round(w * orig.h / orig.w));
  const O = blur(downscale(orig, w, h), 1.0);
  const go = gradient(O);
  const epsO = 0.5 * gradientScale(go.gx, go.gy);
  const nOx = new Float32Array(w * h), nOy = new Float32Array(w * h);
  for (let p = 0; p < w * h; p++) {
    const r = Math.sqrt(go.gx[p] * go.gx[p] + go.gy[p] * go.gy[p] + epsO * epsO);
    nOx[p] = go.gx[p] / r; nOy[p] = go.gy[p] / r;
  }
  const ratio = screenWidthFull(m) * (1 - 2 * m.inset) / w;
  const l = filmLevelFor(ctx, ratio);
  const lv = ctx.filmPyr[l];
  const F = blur(lv, Math.max(0.6, 0.5 * ratio * lv.w / ctx.FW));
  const gF = gradient(F);
  const epsW = 0.5 * gradientScale(gF.gx, gF.gy);
  const n = w * h;
  return {
    w, h, nOx, nOy, epsW, ratio,
    F: F.data, Fw: F.w, Fh: F.h, Fgx: gF.gx, Fgy: gF.gy,
    fx: F.w / ctx.FW, fy: F.h / ctx.FH,
    W: new Float32Array(n), valid: new Uint8Array(n),
    dWX: new Float32Array(n), dWY: new Float32Array(n),
    Xs: new Float32Array(n), Ys: new Float32Array(n), Xt: new Float32Array(n), Yt: new Float32Array(n),
    qx: new Float32Array(n), qy: new Float32Array(n),
    grid: null,
  };
}

// per-column / per-row B-spline weights and envelopes for the current grid
function prepareGrid(L, m) {
  if (L.grid && L.grid.nx === m.nx && L.grid.ny === m.ny) return L.grid;
  const { w, h } = L;
  const colK = new Int32Array(w), colW = new Float64Array(4 * w), colE = new Float64Array(w), colU = new Float64Array(w);
  const rowK = new Int32Array(h), rowW = new Float64Array(4 * h), rowE = new Float64Array(h), rowV = new Float64Array(h);
  for (let i = 0; i < w; i++) {
    const u = (i + 0.5) / w, [k, f] = bsplineIndex(u, m.nx);
    colK[i] = k; bsplineWeights(f, colW, 4 * i); colE[i] = 1; colU[i] = u; // data term fits the band-free content warp
  }
  for (let j = 0; j < h; j++) {
    const v = (j + 0.5) / h, [k, f] = bsplineIndex(v, m.ny);
    rowK[j] = k; bsplineWeights(f, rowW, 4 * j); rowE[j] = 1; rowV[j] = v;
  }
  L.grid = { nx: m.nx, ny: m.ny, colK, colW, colE, colU, rowK, rowW, rowE, rowV };
  return L.grid;
}

// ---------------------------------------------------------------- objective

const _sc = new Float64Array(6);

// Returns energy; fills grad g (same length as p) when given.
// opt: { lambda, pinStrength, region (Float32Array weights or null) }
export function evaluate(L, m, p, g, opt) {
  const G = prepareGrid(L, m);
  const { w, h, nOx, nOy, epsW, F, Fw, Fh, Fgx, Fgy, fx, fy, W, valid, dWX, dWY,
          Xs, Ys, Xt, Yt, qx, qy } = L;
  const nc = m.cx.length, stride = m.nx + 3;
  const ins = m.inset, kk = 1 - 2 * ins, S = m.screen;
  const a0 = p[0], a1 = p[1], a2 = p[2], a3 = p[3], a4 = p[4], a5 = p[5];
  const region = opt.region || null;

  // pass 1: map every canvas pixel into the filmed level, sample value + gradient
  for (let j = 0; j < h; j++) {
    const v = G.rowV[j], ev = G.rowE[j], ky = G.rowK[j], rw = 4 * j, dv = v - 0.5;
    for (let i = 0; i < w; i++) {
      const pix = j * w + i;
      const u = G.colU[i], eu = G.colE[i], kx = G.colK[i], cw = 4 * i, du = u - 0.5;
      let bx = 0, by = 0;
      for (let b = 0; b < 4; b++) {
        const wb = G.rowW[rw + b], base = AFFINE_N + (ky + b) * stride + kx;
        for (let a = 0; a < 4; a++) {
          const wt = G.colW[cw + a] * wb;
          bx += wt * p[base + a]; by += wt * p[base + a + nc];
        }
      }
      const s = u + eu * (0.5 + a0 * du + a1 * dv + a2 - u + bx);
      const t = v + ev * (0.5 + a3 * du + a4 * dv + a5 - v + by);
      const sp = ins + s * kk, tp = ins + t * kk;
      screenEval(S, sp, tp, _sc);
      Xs[pix] = _sc[2] * kk; Ys[pix] = _sc[3] * kk; Xt[pix] = _sc[4] * kk; Yt[pix] = _sc[5] * kk;
      const xl = (_sc[0] + 0.5) * fx - 0.5, yl = (_sc[1] + 0.5) * fy - 0.5;
      // content beyond the lit raster (overscan) has nothing to match against
      if (!(sp > -0.002 && sp < 1.002 && tp > -0.002 && tp < 1.002 &&
            xl >= 0 && yl >= 0 && xl <= Fw - 1.001 && yl <= Fh - 1.001)) {
        valid[pix] = 0; W[pix] = 0; dWX[pix] = 0; dWY[pix] = 0;
        continue;
      }
      const x0 = xl | 0, y0 = yl | 0, tx = xl - x0, ty = yl - y0;
      const q = y0 * Fw + x0;
      const w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
      valid[pix] = 1;
      W[pix] = w00 * F[q] + w10 * F[q + 1] + w01 * F[q + Fw] + w11 * F[q + Fw + 1];
      dWX[pix] = (w00 * Fgx[q] + w10 * Fgx[q + 1] + w01 * Fgx[q + Fw] + w11 * Fgx[q + Fw + 1]) * fx;
      dWY[pix] = (w00 * Fgy[q] + w10 * Fgy[q + 1] + w01 * Fgy[q + Fw] + w11 * Fgy[q + Fw + 1]) * fy;
    }
  }

  // pass 2: NGF similarity of warped vs original
  let Ssum = 0, Z = 0;
  const e2 = epsW * epsW;
  qx.fill(0); qy.fill(0);
  for (let j = 1; j < h - 1; j++) {
    for (let i = 1; i < w - 1; i++) {
      const pix = j * w + i;
      const wt = region ? region[pix] : 1;
      if (wt === 0) continue;
      const ox = nOx[pix], oy = nOy[pix];
      Z += wt * (ox * ox + oy * oy);
      if (!(valid[pix - 1] && valid[pix + 1] && valid[pix - w] && valid[pix + w])) continue;
      const gx = 0.5 * (W[pix + 1] - W[pix - 1]), gy = 0.5 * (W[pix + w] - W[pix - w]);
      const r = Math.sqrt(gx * gx + gy * gy + e2);
      const a = (ox * gx + oy * gy) / r;
      Ssum += wt * a * a;
      if (g) {
        const c = -2 * a / r * wt;
        qx[pix] = c * (ox - a * gx / r);
        qy[pix] = c * (oy - a * gy / r);
      }
    }
  }
  Z = Math.max(Z, 1e-9);
  let E = -Ssum / Z;
  L.lastScore = Ssum / Z;

  if (g) {
    g.fill(0);
    const iZ = 1 / Z;
    // pass 3: adjoint of the finite differences -> force on each pixel's filmed position
    for (let j = 0; j < h; j++) {
      const v = G.rowV[j], ev = G.rowE[j], ky = G.rowK[j], rw = 4 * j, dv = v - 0.5;
      for (let i = 0; i < w; i++) {
        const pix = j * w + i;
        let dEdW = 0;
        if (i > 0) dEdW += 0.5 * qx[pix - 1];
        if (i < w - 1) dEdW -= 0.5 * qx[pix + 1];
        if (j > 0) dEdW += 0.5 * qy[pix - w];
        if (j < h - 1) dEdW -= 0.5 * qy[pix + w];
        if (dEdW === 0 || !valid[pix]) continue;
        dEdW *= iZ;
        const fX = dEdW * dWX[pix], fY = dEdW * dWY[pix];
        const u = G.colU[i], eu = G.colE[i], kx = G.colK[i], cw = 4 * i, du = u - 0.5;
        const gs = (fX * Xs[pix] + fY * Ys[pix]) * eu;
        const gt = (fX * Xt[pix] + fY * Yt[pix]) * ev;
        g[0] += gs * du; g[1] += gs * dv; g[2] += gs;
        g[3] += gt * du; g[4] += gt * dv; g[5] += gt;
        for (let b = 0; b < 4; b++) {
          const wb = G.rowW[rw + b], base = AFFINE_N + (ky + b) * stride + kx;
          for (let a = 0; a < 4; a++) {
            const wt = G.colW[cw + a] * wb;
            g[base + a] += gs * wt; g[base + a + nc] += gt * wt;
          }
        }
      }
    }
  }

  E += regularize(m, p, g, opt.lambda ?? DEFAULTS.lambda);
  if (m.pins.length) E += pinEnergy(L, m, p, g, opt.pinStrength ?? DEFAULTS.pinStrength);
  return E;
}

// Bending energy of the mesh (both components), resolution independent.
function regularize(m, p, g, lambda) {
  if (!lambda) return 0;
  const nx = m.nx, ny = m.ny, sx = nx + 3, sy = ny + 3, nc = m.cx.length;
  const kxx = nx * nx, kyy = ny * ny, kxy = nx * ny;
  const scale = lambda / (nx * ny);
  let E = 0;
  for (let comp = 0; comp < 2; comp++) {
    const off = AFFINE_N + comp * nc;
    for (let l = 0; l < sy; l++) for (let k = 1; k < sx - 1; k++) {
      const i0 = off + l * sx + k;
      const d = (p[i0 - 1] - 2 * p[i0] + p[i0 + 1]) * kxx;
      E += scale * d * d;
      if (g) { const c = 2 * scale * d * kxx; g[i0 - 1] += c; g[i0] -= 2 * c; g[i0 + 1] += c; }
    }
    for (let l = 1; l < sy - 1; l++) for (let k = 0; k < sx; k++) {
      const i0 = off + l * sx + k;
      const d = (p[i0 - sx] - 2 * p[i0] + p[i0 + sx]) * kyy;
      E += scale * d * d;
      if (g) { const c = 2 * scale * d * kyy; g[i0 - sx] += c; g[i0] -= 2 * c; g[i0 + sx] += c; }
    }
    for (let l = 0; l < sy - 1; l++) for (let k = 0; k < sx - 1; k++) {
      const i0 = off + l * sx + k;
      const d = (p[i0 + sx + 1] - p[i0 + 1] - p[i0 + sx] + p[i0]) * kxy;
      E += 2 * scale * d * d;
      if (g) {
        const c = 4 * scale * d * kxy;
        g[i0 + sx + 1] += c; g[i0 + 1] -= c; g[i0 + sx] -= c; g[i0] += c;
      }
    }
  }
  return E;
}

const _wu = new Float64Array(4), _wv = new Float64Array(4);

// User pins: canvas point (u,v) must land on filmed point (X,Y).
function pinEnergy(L, m, p, g, mu) {
  const nc = m.cx.length, stride = m.nx + 3, ins = m.inset, kk = 1 - 2 * ins;
  const sig2 = L.ratio * L.ratio;
  let E = 0;
  for (const pin of m.pins) {
    const { u, v } = pin;
    const [kx, fu] = bsplineIndex(u, m.nx), [ky, fv] = bsplineIndex(v, m.ny);
    bsplineWeights(fu, _wu); bsplineWeights(fv, _wv);
    let bx = 0, by = 0;
    for (let b = 0; b < 4; b++) for (let a = 0; a < 4; a++) {
      const c = AFFINE_N + (ky + b) * stride + kx + a, wt = _wu[a] * _wv[b];
      bx += wt * p[c]; by += wt * p[c + nc];
    }
    const du = u - 0.5, dv = v - 0.5, eu = envelope(u, m.margin[0]), ev = envelope(v, m.margin[1]);
    const s = u + eu * (0.5 + p[0] * du + p[1] * dv + p[2] - u + bx);
    const t = v + ev * (0.5 + p[3] * du + p[4] * dv + p[5] - v + by);
    screenEval(m.screen, ins + s * kk, ins + t * kk, _sc);
    const rX = _sc[0] - pin.X, rY = _sc[1] - pin.Y;
    E += mu * (rX * rX + rY * rY) / sig2;
    if (g) {
      const fX = 2 * mu * rX / sig2, fY = 2 * mu * rY / sig2;
      const gs = (fX * _sc[2] + fY * _sc[3]) * kk * eu, gt = (fX * _sc[4] + fY * _sc[5]) * kk * ev;
      g[0] += gs * du; g[1] += gs * dv; g[2] += gs;
      g[3] += gt * du; g[4] += gt * dv; g[5] += gt;
      for (let b = 0; b < 4; b++) for (let a = 0; a < 4; a++) {
        const c = AFFINE_N + (ky + b) * stride + kx + a, wt = _wu[a] * _wv[b];
        g[c] += gs * wt; g[c + nc] += gt * wt;
      }
    }
  }
  return E;
}

// ---------------------------------------------------------------- L-BFGS

function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

export async function lbfgs(f, x0, { maxIter = 30, mem = 8, freeze = null, stepPx = 0.5,
                                     onIter = null } = {}) {
  const n = x0.length;
  let x = Float64Array.from(x0), g = new Float64Array(n);
  let fx = f(x, g);
  const mask = v => { if (freeze) for (let i = 0; i < n; i++) if (freeze[i]) v[i] = 0; return v; };
  mask(g);
  const S = [], Y = [], R = [];
  let stall = 0;
  for (let it = 0; it < maxIter; it++) {
    const q = Float64Array.from(g), alpha = new Float64Array(S.length);
    for (let i = S.length - 1; i >= 0; i--) {
      alpha[i] = R[i] * dot(S[i], q);
      for (let k = 0; k < n; k++) q[k] -= alpha[i] * Y[i][k];
    }
    let gamma;
    if (S.length) gamma = dot(S[S.length - 1], Y[Y.length - 1]) / dot(Y[Y.length - 1], Y[Y.length - 1]);
    else { let mx = 0; for (let k = 0; k < n; k++) mx = Math.max(mx, Math.abs(g[k])); gamma = stepPx / Math.max(mx, 1e-12); }
    for (let k = 0; k < n; k++) q[k] *= gamma;
    for (let i = 0; i < S.length; i++) {
      const beta = R[i] * dot(Y[i], q);
      for (let k = 0; k < n; k++) q[k] += S[i][k] * (alpha[i] - beta);
    }
    const d = mask(q.map(v => -v));
    let gd = dot(g, d);
    if (!(gd < 0)) {
      S.length = Y.length = R.length = 0;
      let mx = 0; for (let k = 0; k < n; k++) mx = Math.max(mx, Math.abs(g[k]));
      const sc = stepPx / Math.max(mx, 1e-12);
      for (let k = 0; k < n; k++) d[k] = -g[k] * sc;
      mask(d); gd = dot(g, d);
      if (!(gd < 0)) break;
    }
    let step = 1, xn = new Float64Array(n), gn = new Float64Array(n), fn = Infinity, ok = false;
    for (let ls = 0; ls < 14; ls++) {
      for (let k = 0; k < n; k++) xn[k] = x[k] + step * d[k];
      fn = f(xn, gn);
      if (fn <= fx + 1e-4 * step * gd) { ok = true; break; }
      step *= 0.5;
    }
    if (!ok) break;
    mask(gn);
    const s = new Float64Array(n), y = new Float64Array(n);
    for (let k = 0; k < n; k++) { s[k] = xn[k] - x[k]; y[k] = gn[k] - g[k]; }
    const sy = dot(s, y);
    if (sy > 1e-16) { S.push(s); Y.push(y); R.push(1 / sy); if (S.length > mem) { S.shift(); Y.shift(); R.shift(); } }
    const rel = (fx - fn) / Math.max(1e-9, Math.abs(fx));
    x = xn; g = gn; fx = fn;
    stall = rel < 1e-5 ? stall + 1 : 0;
    if (onIter) await onIter(it, fx);
    if (stall >= 3) break;
  }
  return { x, f: fx };
}

// ---------------------------------------------------------------- global search

// Brute-force search of the residual content zoom/offset on a tiny canvas, using a
// pre-rectified screen image so each candidate is just a resample + NGF score.
export function searchAffine(ctx, m, keep = 4) {
  const cw = 40, ch = Math.max(8, Math.round(cw * ctx.orig.h / ctx.orig.w));
  const O = blur(downscale(ctx.orig, cw, ch), 0.8);
  const go = gradient(O);
  const epsO = 0.5 * gradientScale(go.gx, go.gy);
  // rectified screen: R(s,t) over s,t in [lo, hi]
  const lo = -0.4, hi = 1.4, rw = 3 * cw, rh = 3 * ch;
  const ins = m.inset, kk = 1 - 2 * ins;
  const ratio = screenWidthFull(m) * kk * (hi - lo) / rw;
  const lv = ctx.filmPyr[filmLevelFor(ctx, ratio)];
  const F = blur(lv, Math.max(0.6, 0.5 * ratio * lv.w / ctx.FW));
  const fx = F.w / ctx.FW, fy = F.h / ctx.FH;
  const Rimg = new Float32Array(rw * rh);
  for (let j = 0; j < rh; j++) for (let i = 0; i < rw; i++) {
    const s = lo + (hi - lo) * (i + 0.5) / rw, t = lo + (hi - lo) * (j + 0.5) / rh;
    screenEval(m.screen, ins + s * kk, ins + t * kk, _sc);
    let xl = (_sc[0] + 0.5) * fx - 0.5, yl = (_sc[1] + 0.5) * fy - 0.5;
    if (!(xl >= 0 && yl >= 0 && xl < F.w - 1 && yl < F.h - 1)) { Rimg[j * rw + i] = 0; continue; }
    const x0 = xl | 0, y0 = yl | 0, tx = xl - x0, ty = yl - y0, q = y0 * F.w + x0, d = F.data;
    Rimg[j * rw + i] = (1 - tx) * (1 - ty) * d[q] + tx * (1 - ty) * d[q + 1] +
                       (1 - tx) * ty * d[q + F.w] + tx * ty * d[q + F.w + 1];
  }
  const R = blur({ data: Rimg, w: rw, h: rh }, 1.0).data;
  const gR = gradient({ data: R, w: rw, h: rh });
  const epsW = 0.5 * gradientScale(gR.gx, gR.gy);

  const m0 = Math.max(m.margin[0], m.margin[1], 0.08);
  const Wv = new Float32Array(cw * ch);
  // interior pixels and their normalized original edges, fixed for all candidates
  const pix = [], oxs = [], oys = [];
  let Z = 0;
  for (let j = 1; j < ch - 1; j++) {
    const v = (j + 0.5) / ch;
    if (v < m0 || v > 1 - m0) continue;
    for (let i = 1; i < cw - 1; i++) {
      const u = (i + 0.5) / cw;
      if (u < m0 || u > 1 - m0) continue;
      const p = j * cw + i;
      const r0 = Math.sqrt(go.gx[p] * go.gx[p] + go.gy[p] * go.gy[p] + epsO * epsO);
      pix.push(p); oxs.push(go.gx[p] / r0); oys.push(go.gy[p] / r0);
      Z += (go.gx[p] * go.gx[p] + go.gy[p] * go.gy[p]) / (r0 * r0);
    }
  }
  const e2 = epsW * epsW;
  const score = (sx, sy, tx, ty) => {
    for (let j = 0; j < ch; j++) {
      const v = (j + 0.5) / ch, t = 0.5 + sy * (v - 0.5) + ty;
      const yr = (t - lo) / (hi - lo) * rh - 0.5;
      for (let i = 0; i < cw; i++) {
        const u = (i + 0.5) / cw, s = 0.5 + sx * (u - 0.5) + tx;
        const xr = (s - lo) / (hi - lo) * rw - 0.5;
        if (!(xr >= 0 && yr >= 0 && xr < rw - 1 && yr < rh - 1)) { Wv[j * cw + i] = 0; continue; }
        const x0 = xr | 0, y0 = yr | 0, ax = xr - x0, ay = yr - y0, q = y0 * rw + x0;
        Wv[j * cw + i] = (1 - ax) * (1 - ay) * R[q] + ax * (1 - ay) * R[q + 1] +
                         (1 - ax) * ay * R[q + rw] + ax * ay * R[q + rw + 1];
      }
    }
    let S = 0;
    for (let k = 0; k < pix.length; k++) {
      const p = pix[k];
      const gx = 0.5 * (Wv[p + 1] - Wv[p - 1]), gy = 0.5 * (Wv[p + cw] - Wv[p - cw]);
      const a = (oxs[k] * gx + oys[k] * gy) / Math.sqrt(gx * gx + gy * gy + e2);
      S += a * a;
    }
    return S / Math.max(Z, 1e-9);
  };

  const cands = [];
  for (let sx = 0.8; sx <= 1.5001; sx += 0.05)
    for (let sy = 0.8; sy <= 1.5001; sy += 0.05)
      for (let tx = -0.2; tx <= 0.2001; tx += 1 / cw)
        for (let ty = -0.2; ty <= 0.2001; ty += 1 / ch) {
          cands.push([score(sx, sy, tx, ty), sx, sy, tx, ty]);
        }
  cands.sort((a, b) => b[0] - a[0]);
  // keep distinct top candidates
  const top = [];
  for (const c of cands) {
    if (top.every(t => Math.abs(t[1] - c[1]) > 0.06 || Math.abs(t[2] - c[2]) > 0.06 ||
                       Math.abs(t[3] - c[3]) > 0.04 || Math.abs(t[4] - c[4]) > 0.04)) top.push(c);
    if (top.length >= keep) break;
  }
  return top;
}

// ---------------------------------------------------------------- pipeline

export function makeCtx(orig, film, FW, FH) {
  return { orig, film, FW, FH, filmPyr: pyramid(film, 24) };
}

function scheduleFor(quality) {
  return QUALITY[quality] || QUALITY.medium;
}

export async function runLevel(ctx, m, spec, opt, report, label) {
  const g = gridForAspect(spec.n, ctx.orig.w, ctx.orig.h);
  refitGrid(m, g.nx, g.ny);
  const L = buildLevel(ctx, m, spec.w);
  let freeze = null;
  if (spec.affineOnly) {
    freeze = new Uint8Array(6 + 2 * m.cx.length).fill(1).fill(0, 0, 6);
    if (spec.noShear) { freeze[1] = 1; freeze[3] = 1; }
  }
  const res = await lbfgs((x, gr) => evaluate(L, m, x, gr, opt), getParams(m), {
    maxIter: spec.it, freeze, stepPx: 0.5 / L.w,
    onIter: async (it, f) => { if (report) report({ stage: label, iter: it, of: spec.it, energy: f }); await tick(); },
  });
  setParams(m, res.x);
  evaluate(L, m, res.x, null, opt);
  return { score: L.lastScore, energy: res.f, level: L };
}

// Full automatic alignment.
//   orig: luma {data,w,h} of the original (any resolution, correct aspect)
//   film: luma {data,w,h} of the filmed photo/frame (any resolution)
//   W,H: original full size; FW,FH: filmed full size (model coordinates)
export async function align({ orig, film, W, H, FW, FH, screen = null, quality = 'medium',
                              lambda = DEFAULTS.lambda, margin = DEFAULTS.margin,
                              onProgress = null }) {
  const report = onProgress || (() => {});
  report({ stage: 'screen' });
  let scr = screen;
  if (!scr) {
    const d = detectScreen(film);
    const k = FW / film.w;
    scr = { corners: d.corners.map(p => [p[0] * k, p[1] * k]), bulge: d.bulge.map(b => b * k),
            inset: d.inset, detected: d.ok };
  }
  const m = createModel({ W, H, FW, FH, corners: scr.corners, bulge: scr.bulge,
                          inset: scr.inset ?? 0.02, margin: margin === 'auto' ? 0.05 : margin,
                          nx: 2, ny: 2 });
  const ctx = makeCtx(orig, film, FW, FH);
  const opt = { lambda };

  report({ stage: 'search' });
  await tick();
  // The 40 px brute-force landscape is flat for busy textures and coarse in offset, so
  // its winners are mixed with a regular grid of zoom seeds; every start is polished
  // cheaply, the best are polished again at higher resolution.
  const seeds = searchAffine(ctx, m, 6).map(([, sx, sy, tx, ty]) => [sx, 0, tx, 0, sy, ty]);
  for (const sx of [1.0, 1.12, 1.25, 1.4]) for (const sy of [1.0, 1.12, 1.25, 1.4]) seeds.push([sx, 0, 0, 0, sy, 0]);
  const same = (a, b) => Math.abs(a[0] - b[0]) < 0.012 && Math.abs(a[4] - b[4]) < 0.012 &&
                         Math.abs(a[2] - b[2]) < 0.006 && Math.abs(a[5] - b[5]) < 0.006;
  const polishAll = async (models, spec) => {
    const out = [];
    for (const mm of models) {
      const r = await runLevel(ctx, mm, spec, opt, null, 'search');
      if (out.some(q => same(q.model.affine, mm.affine))) continue;
      out.push({ score: r.score, model: mm });
    }
    return out.sort((x, y) => y.score - x.score);
  };
  let polished = await polishAll(seeds.map(a => { const mm = cloneModel(m); mm.affine.set(a); return mm; }),
                                 { w: 64, n: 2, it: 15, affineOnly: true, noShear: true });
  // keep the best candidate of each distinct zoom: coarse scores of different zooms
  // are often within noise of each other, the mesh tournament below decides
  const diverse = (list, n) => {
    const out = [];
    for (const q of list) {
      if (out.every(o => Math.abs(o.model.affine[0] - q.model.affine[0]) > 0.05 ||
                         Math.abs(o.model.affine[4] - q.model.affine[4]) > 0.05)) out.push(q);
      if (out.length >= n) break;
    }
    return out;
  };
  report({ stage: 'search', iter: 1, of: 2 });
  polished = await polishAll(diverse(polished, 6).map(q => q.model),
                             { w: 128, n: 2, it: 12, affineOnly: true, noShear: true });
  report({ stage: 'search', iter: 2, of: 2, candidates: polished.slice(0, 5).map(q => [q.score, ...q.model.affine]) });

  // tournament: the best few candidates each run the first levels with the mesh;
  // repetitive textures (spots, windows) can fool the tiny search canvas, but rarely
  // survive a 256 px comparison
  const sched = scheduleFor(quality);
  const pre = Math.min(2, sched.length);
  const finalists = diverse(polished, 4);
  let winner = null;
  for (let c = 0; c < finalists.length; c++) {
    let r = null;
    for (let i = 0; i < pre; i++) {
      r = await runLevel(ctx, finalists[c].model, sched[i], opt, report,
                         `candidate ${c + 1}/${finalists.length}, level ${i + 1}/${sched.length}`);
    }
    report({ stage: 'tournament', candidate: c, score: r.score, affine: Array.from(finalists[c].model.affine) });
    if (!winner || r.score > winner.score) winner = { score: r.score, model: finalists[c].model };
  }
  const best = winner.model;
  let last = { score: winner.score };
  for (let i = pre; i < sched.length; i++) {
    last = await runLevel(ctx, best, sched[i], opt, report, `level ${i + 1}/${sched.length}`);
    report({ stage: `level ${i + 1}/${sched.length}`, score: last.score });
  }
  if (margin === 'auto') fitMargins(best);
  return { model: best, score: last.score, screenDetected: scr.detected !== false };
}

// Re-solve from the current model (after pins or screen edits); keeps the global fit.
export async function refine({ model, orig, film, quality = 'medium', lambda = DEFAULTS.lambda,
                               pinStrength = DEFAULTS.pinStrength, onProgress = null,
                               fromLevel = null, autoMargin = true }) {
  const ctx = makeCtx(orig, film, model.FW, model.FH);
  const opt = { lambda, pinStrength };
  const sched = scheduleFor(quality);
  const start = fromLevel ?? Math.max(0, sched.length - 2);
  let last = null;
  for (let i = start; i < sched.length; i++) {
    last = await runLevel(ctx, model, sched[i], opt, onProgress, `refine ${i + 1}/${sched.length}`);
  }
  if (autoMargin) fitMargins(model);
  return { model, score: last.score };
}

// Score of an existing model at a given canvas width (for reporting / comparisons).
export function scoreModel({ model, orig, film, w = 512 }) {
  const ctx = makeCtx(orig, film, model.FW, model.FH);
  const L = buildLevel(ctx, model, w);
  evaluate(L, model, getParams(model), null, { lambda: 0 });
  return L.lastScore;
}

export { makeScreen };
