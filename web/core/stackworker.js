// Stack-mode worker. Several run in parallel (one image each); the reference is
// prepared once per worker and cached.
import { prepare, alignPair, brightestSpot } from './stack.js';

let refKey = null, refPrep = null;

const prep = (l, mask) => prepare({ data: l.data, w: l.w, h: l.h }, l.W, l.H, { mask });

self.onmessage = ({ data: msg }) => {
  const { id, kind, args } = msg;
  try {
    if (kind === 'ref') {
      refKey = args.key;
      refPrep = prep(args.lum);
      self.postMessage({ id, type: 'done' });
    } else if (kind === 'pair') {
      // align args.lum (target) to the cached reference, or to args.base if given
      const base = args.base ? prep(args.base) : refPrep;
      if (!base || (!args.base && args.key !== refKey)) throw new Error('reference not ready');
      const tgt = prep(args.lum);
      const r = alignPair(base, tgt, { kind: args.kind, roi: args.roi, anchor: args.anchor });
      self.postMessage({ id, type: 'done', result: r });
    } else if (kind === 'spot') {
      const s = brightestSpot({ data: args.lum.data, w: args.lum.w, h: args.lum.h }, args.lum.W, args.lum.H, args.near, args.radius);
      self.postMessage({ id, type: 'done', result: s });
    }
  } catch (err) {
    self.postMessage({ id, type: 'error', message: err && err.message ? err.message : String(err) });
  }
};
