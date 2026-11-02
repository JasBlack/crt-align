// Temporal sync between an original video and a filmed copy of it.
// Signatures are per-frame mean luma of the centre crop; both are resampled to a
// common rate so different frame rates line up.

const RATE = 25;

function resample(sig, fps) {
  const n = sig.length;
  if (n < 2) return Float64Array.from(sig);
  const dur = (n - 1) / fps, m = Math.max(2, Math.floor(dur * RATE) + 1);
  const out = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    const x = (i / RATE) * fps, i0 = Math.min(n - 2, Math.floor(x)), t = x - i0;
    out[i] = sig[i0] * (1 - t) + sig[i0 + 1] * t;
  }
  return out;
}

function standardize(a) {
  let m = 0;
  for (const v of a) m += v;
  m /= a.length;
  let s = 0;
  for (const v of a) s += (v - m) * (v - m);
  s = Math.sqrt(s / a.length) || 1;
  return a.map(v => (v - m) / s);
}

// Candidate offsets o (seconds): filmed time t + o shows original time t.
// Returns up to k peaks, strongest first, at least 1 s apart.
export function syncCandidates(sigRef, fpsRef, sigMov, fpsMov, k = 5) {
  const a = standardize(resample(sigRef, fpsRef)), b = standardize(resample(sigMov, fpsMov));
  const na = a.length, nb = b.length, scores = [];
  for (let lag = -(na - 1); lag <= nb - 1; lag++) {
    // overlap of a[i] with b[i + lag]
    const i0 = Math.max(0, -lag), i1 = Math.min(na, nb - lag);
    const n = i1 - i0;
    if (n < Math.min(na, nb) * 0.3 || n < RATE) continue;
    let s = 0;
    for (let i = i0; i < i1; i++) s += a[i] * b[i + lag];
    scores.push([s / n, lag]);
  }
  scores.sort((x, y) => y[0] - x[0]);
  const picks = [];
  for (const [s, lag] of scores) {
    if (picks.every(p => Math.abs(p.lag - lag) > RATE)) picks.push({ lag, corr: s });
    if (picks.length >= k) break;
  }
  return picks.map(p => ({ offset: p.lag / RATE, corr: p.corr }));
}
