import { createEngine } from './runtime.mjs';

const maxPhotoBytes = 64 * 1024 * 1024;
const maxFeatureBytes = 16 * 1024;
const runtime = createEngine();
let models;
let queue = Promise.resolve();

async function readPhoto(api, file, score, extract) {
  if (!file || file.size === 0 || file.size > maxPhotoBytes)
    throw new Error('请选择有效照片，单张文件上限为 64 MiB。');
  return api.analyzePhoto(file, Number(score), extract);
}

async function featureText(file) {
  if (!file || file.size === 0 || file.size > maxFeatureBytes)
    throw new Error('请选择有效的特征 JSON，单个文件上限为 16 KiB。');
  return file.text();
}

function comparison(similarity, threshold) {
  const differentThreshold = Math.min(.363, threshold);
  const isMatch = similarity >= threshold;
  return { similarity, threshold, differentThreshold, isMatch,
    decision: isMatch ? 'same' : similarity < differentThreshold ? 'different' : 'uncertain' };
}

async function analyze(api, mode, first, second, options) {
  const threshold = options.threshold === '' ? .68 : Number(options.threshold);
  if (mode === 'vectors') {
    const a = await featureText(first);
    const b = await featureText(second);
    const start = performance.now();
    const similarity = api.compareVectors(a, b);
    return { status: 200, ok: true, result: { backend: 'wasm', deviceName: api.name,
      elapsedMs: Math.round(performance.now() - start), comparison: comparison(similarity, threshold) } };
  }

  try {
    await (models ??= api.loadModels());
  } catch (error) {
    models = undefined;
    throw error;
  }
  const a = await readPhoto(api, first, options.score, mode !== 'detect');
  const b = mode === 'compare' ? await readPhoto(api, second, options.score, true) : null;
  const images = (b ? [a, b] : [a]).map(({ width, height, faces }) => ({ width, height, faces }));
  const result = { backend: 'wasm', deviceName: api.name, images, elapsedMs: a.elapsedMs + (b?.elapsedMs ?? 0) };
  if (mode === 'detect') return { status: 200, ok: true, result };
  if (!a.feature || (b && !b.feature)) {
    result.message = b
      ? '比对要求每张图恰好一张脸。照片 A 检出 ' + a.faces.length + ' 张，照片 B 检出 ' + b.faces.length + ' 张。'
      : '提取特征要求恰好一张脸，检出了 ' + a.faces.length + ' 张。';
    return { status: 422, ok: false, result };
  }
  if (b) {
    const similarity = api.compareVectors(JSON.stringify(a.feature), JSON.stringify(b.feature));
    result.comparison = comparison(similarity, threshold);
  } else {
    result.feature = a.feature;
  }
  return { status: 200, ok: true, result };
}

self.onmessage = ({ data: { id, mode, first, second, options } }) => {
  queue = queue.then(async () => {
    try {
      const api = await runtime;
      const value = mode === 'init' ? api.name : await analyze(api, mode, first, second, options);
      self.postMessage({ id, value });
    } catch (error) {
      self.postMessage({ id, error: error.message });
    }
  });
};
