// Loading images / videos, luma extraction, and saving files.

import { lumaFromRGBA, downscale } from './core/image.js';

const MEDIABUNNY = 'https://cdn.jsdelivr.net/npm/mediabunny@1.61.4/dist/bundles/mediabunny.min.mjs';
let mbPromise = null;
export const mb = () => (mbPromise ??= import(MEDIABUNNY));

const VIDEO_EXT = /\.(mp4|mov|m4v|webm|mkv)$/i;
export const isVideoFile = f => (f.type || '').startsWith('video/') || VIDEO_EXT.test(f.name || '');

// Estimation working sizes (see core/optimize.js levels; the original's finest
// level is 768 px, the filmed photo is sampled at roughly 2x the canvas spacing).
const ORIG_EST = 1536, FILM_EST = 2048;

function lumaOf(source, w, h, maxW) {
  // read pixels at full size, then our anti-aliased downscale (scanline-safe)
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, w, h);
  const img = lumaFromRGBA(ctx.getImageData(0, 0, w, h).data, w, h);
  if (w <= maxW) return img;
  return downscale(img, maxW, Math.max(1, Math.round(h * maxW / w)));
}

export async function loadImage(file, role) {
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const w = bitmap.width, h = bitmap.height;
  return {
    kind: 'image', name: file.name, file, w, h, bitmap,
    lum: lumaOf(bitmap, w, h, role === 'orig' ? ORIG_EST : FILM_EST),
    frame: async () => bitmap,
  };
}

export async function loadVideo(file) {
  const M = await mb();
  const input = new M.Input({ source: new M.BlobSource(file), formats: M.ALL_FORMATS });
  const track = await input.getPrimaryVideoTrack();
  if (!track) throw new Error(`${file.name} has no video track.`);
  if (!(await track.canDecode())) {
    throw new Error(`This browser can't decode the video in ${file.name} (codec ${track.codec || 'unknown'}). Try Chrome or Edge, or re-encode to H.264 MP4.`);
  }
  const start = await input.getFirstTimestamp([track]);
  const duration = (await input.computeDuration([track])) - start;
  let fps = 30;
  try { fps = (await track.computeFrameRateMetrics()).bestGuessFrameRate || 30; } catch { /* keep 30 */ }
  const w = track.displayWidth, h = track.displayHeight;
  const sink = new M.CanvasSink(track, { poolSize: 4 });
  const v = {
    kind: 'video', name: file.name, file, w, h, fps, duration, start, track, sink, input,
    // frame at time t (seconds from the start of the clip)
    async frame(t) {
      const c = await sink.getCanvas(start + Math.max(0, Math.min(duration - 1e-3, t)));
      return c ? c.canvas : null;
    },
  };
  return v;
}

export async function lumaAt(media, t, role) {
  const src = await media.frame(t);
  if (!src) throw new Error(`Couldn't decode a frame of ${media.name} at ${t.toFixed(2)} s.`);
  return lumaOf(src, media.w, media.h, role === 'orig' ? ORIG_EST : FILM_EST);
}

// Per-frame mean luma of the centre crop, for temporal sync.
export async function lumaSignature(video, onProgress) {
  const M = await mb();
  const sink = new M.CanvasSink(video.track, { width: 64, height: 36, fit: 'fill', poolSize: 2 });
  const sig = [];
  let n = 0;
  const total = Math.max(1, Math.round(video.duration * video.fps));
  for await (const wc of sink.canvases()) {
    const ctx = wc.canvas.getContext('2d', { willReadFrequently: true });
    const d = ctx.getImageData(16, 9, 32, 18).data;
    let s = 0;
    for (let i = 0; i < d.length; i += 4) s += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    sig.push(s / (d.length / 4));
    if (onProgress && ++n % 30 === 0) onProgress(n / total);
  }
  return sig;
}

// ---------------------------------------------------------------- saving

let downloadsPromise = null;
export function prepareDownloads() {
  if (!downloadsPromise) {
    downloadsPromise = (window.claude && typeof window.claude.use === 'function')
      ? window.claude.use('downloads').catch(() => null)
      : Promise.resolve(null);
  }
  return downloadsPromise;
}

// Returns 'saved' | 'declined'. Inside a claude.ai artifact the viewer confirms the
// save; anywhere else this is a normal browser download.
export async function saveFile(filename, blob) {
  const dl = await prepareDownloads();
  if (dl) {
    try {
      await dl.save({ filename, data: blob });
      return 'saved';
    } catch (e) {
      if (e && e.code === 'declined') return 'declined';
      if (e && e.code === 'rate_limited') throw new Error('A save prompt is already open. Answer it, then try again.');
      if (!(e && ['unavailable', 'not_granted', 'capability_disabled', 'capability_removed'].includes(e.code))) {
        throw new Error(e?.message || 'The save failed.');
      }
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  return 'saved';
}
