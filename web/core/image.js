// Grayscale float image helpers. Images are { data: Float32Array, w, h }.

export function lumaFromRGBA(rgba, w, h) {
  const out = new Float32Array(w * h);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = (0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2]) * (1 / 255);
  }
  return { data: out, w, h };
}

// Per-destination-index source spans with coverage weights (area resampling).
function areaSpans(sn, dn) {
  const f = sn / dn;
  const spans = [];
  for (let d = 0; d < dn; d++) {
    let a = d * f, b = a + f;
    if (f < 1) {
      // upscaling: fall back to linear interpolation between neighbours
      const c = (d + 0.5) * f - 0.5;
      const i0 = Math.max(0, Math.min(sn - 1, Math.floor(c)));
      const i1 = Math.min(sn - 1, i0 + 1);
      const t = Math.max(0, Math.min(1, c - i0));
      spans.push({ start: i0, w: i1 > i0 ? [1 - t, t] : [1] });
      continue;
    }
    const i0 = Math.floor(a), i1 = Math.min(sn - 1, Math.ceil(b) - 1);
    const w = [];
    for (let i = i0; i <= i1; i++) {
      w.push((Math.min(b, i + 1) - Math.max(a, i)) / f);
    }
    spans.push({ start: i0, w });
  }
  return spans;
}

export function resize(img, dw, dh) {
  const { data: src, w: sw, h: sh } = img;
  if (sw === dw && sh === dh) return { data: src.slice(), w: dw, h: dh };
  const hs = areaSpans(sw, dw), vs = areaSpans(sh, dh);
  const tmp = new Float32Array(dw * sh);
  for (let y = 0; y < sh; y++) {
    const row = y * sw, orow = y * dw;
    for (let x = 0; x < dw; x++) {
      const { start, w } = hs[x];
      let acc = 0;
      for (let k = 0; k < w.length; k++) acc += src[row + start + k] * w[k];
      tmp[orow + x] = acc;
    }
  }
  const out = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const { start, w } = vs[y];
    const orow = y * dw;
    for (let k = 0; k < w.length; k++) {
      const wk = w[k], srow = (start + k) * dw;
      for (let x = 0; x < dw; x++) out[orow + x] += tmp[srow + x] * wk;
    }
  }
  return { data: out, w: dw, h: dh };
}

export function blur(img, sigma) {
  if (sigma < 0.3) return { data: img.data.slice(), w: img.w, h: img.h };
  const { data, w, h } = img;
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float32Array(2 * r + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) s += (k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma)));
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        const xx = x + i < 0 ? 0 : x + i >= w ? w - 1 : x + i;
        acc += data[row + xx] * k[i + r];
      }
      tmp[row + x] = acc;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        const yy = y + i < 0 ? 0 : y + i >= h ? h - 1 : y + i;
        acc += tmp[yy * w + x] * k[i + r];
      }
      out[y * w + x] = acc;
    }
  }
  return { data: out, w, h };
}

// Central-difference gradient; zero on the 1-pixel border.
export function gradient(img) {
  const { data, w, h } = img;
  const gx = new Float32Array(w * h), gy = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const p = y * w + x;
      gx[p] = 0.5 * (data[p + 1] - data[p - 1]);
      gy[p] = 0.5 * (data[p + w] - data[p - w]);
    }
  }
  return { gx, gy };
}

// Anti-aliased downscale: Gaussian pre-filter, then area resample. CRT scanlines and
// shadow-mask texture sit near Nyquist; a plain box filter folds them into fake
// low-frequency moire bands that the edge matcher would happily lock onto.
export function downscale(img, dw, dh) {
  const f = Math.max(img.w / dw, img.h / dh);
  const src = f > 1.2 ? blur(img, 0.6 * f) : img;
  return resize(src, dw, dh);
}

// Halving pyramid; level 0 is the input. Each level records its size.
export function pyramid(img, minSide = 48) {
  const levels = [img];
  let cur = img;
  while (Math.min(cur.w, cur.h) / 2 >= minSide) {
    cur = downscale(cur, Math.max(1, Math.round(cur.w / 2)), Math.max(1, Math.round(cur.h / 2)));
    levels.push(cur);
  }
  return levels;
}

export function median(values) {
  const a = Float32Array.from(values).sort();
  return a.length ? a[a.length >> 1] : 0;
}

// Robust gradient scale: median magnitude of a strided subsample.
export function gradientScale(gx, gy, mask = null) {
  const vals = [];
  const step = Math.max(1, Math.floor(gx.length / 20000));
  for (let i = 0; i < gx.length; i += step) {
    if (mask && !mask[i]) continue;
    vals.push(Math.hypot(gx[i], gy[i]));
  }
  return Math.max(1e-4, median(vals));
}

export function otsu(data) {
  const hist = new Float64Array(256);
  for (let i = 0; i < data.length; i++) {
    hist[Math.max(0, Math.min(255, Math.round(data[i] * 255)))]++;
  }
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0, wB = 0, best = 0, thr = 0.5;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (!wB) continue;
    const wF = data.length - wB;
    if (!wF) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const v = wB * wF * (mB - mF) * (mB - mF);
    if (v > best) { best = v; thr = t / 255; }
  }
  return thr;
}
