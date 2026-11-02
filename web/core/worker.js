// Runs the aligner off the main thread: the UI stays responsive, a hidden tab
// doesn't throttle it, and terminating the worker cancels a solve.
import { align, refine, scoreModel } from './optimize.js';
import { modelToJSON, modelFromJSON } from './model.js';

self.onmessage = async ({ data: msg }) => {
  const { id, kind, args } = msg;
  const onProgress = e => self.postMessage({ id, type: 'progress', e });
  try {
    if (kind === 'score') {
      const score = scoreModel({ ...args, model: modelFromJSON(args.model) });
      self.postMessage({ id, type: 'done', score });
      return;
    }
    const res = kind === 'align'
      ? await align({ ...args, onProgress })
      : await refine({ ...args, model: modelFromJSON(args.model), onProgress });
    self.postMessage({ id, type: 'done', model: modelToJSON(res.model), score: res.score,
                       screenDetected: res.screenDetected !== false });
  } catch (err) {
    self.postMessage({ id, type: 'error', message: err && err.message ? err.message : String(err) });
  }
};
