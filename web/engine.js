// Main-thread client for core/worker.js.
import { modelFromJSON, modelToJSON } from './core/model.js';

let worker = null, nextId = 1;
const pending = new Map();

function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./core/worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data }) => {
    const p = pending.get(data.id);
    if (!p) return;
    if (data.type === 'progress') { p.onProgress?.(data.e); return; }
    pending.delete(data.id);
    if (data.type === 'error') p.reject(new Error(data.message));
    else p.resolve({ ...data, model: data.model ? modelFromJSON(data.model) : undefined });
  };
  worker.onerror = e => {
    for (const p of pending.values()) p.reject(new Error(e.message || 'The alignment worker crashed.'));
    pending.clear();
    worker = null;
  };
  return worker;
}

function call(kind, args, onProgress) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject, onProgress });
    getWorker().postMessage({ id, kind, args });
  });
}

const lum = l => ({ data: l.data, w: l.w, h: l.h });

export const runAlign = (args, onProgress) =>
  call('align', { ...args, orig: lum(args.orig), film: lum(args.film) }, onProgress);
export const runRefine = (args, onProgress) =>
  call('refine', { ...args, orig: lum(args.orig), film: lum(args.film), model: modelToJSON(args.model) }, onProgress);
export const runScore = args =>
  call('score', { ...args, orig: lum(args.orig), film: lum(args.film), model: modelToJSON(args.model) });

// Stop whatever is running; pending calls reject with 'cancelled'.
export function cancelEngine() {
  if (!worker) return;
  worker.terminate();
  worker = null;
  for (const p of pending.values()) p.reject(new Error('cancelled'));
  pending.clear();
}
