import { server } from './host-config.js';
import { analyzeServer } from './server-api.js';

export { server };
export const local = true;

const worker = new Worker(new URL('face-worker.mjs', import.meta.url), { type: 'module' });
const pending = new Map();
let sequence = 0;
let failure;

worker.onmessage = ({ data }) => pending.get(data.id)?.(data);
worker.onerror = () => {
  failure = '无法启动 WASM，请用 HTTP 服务打开完整的浏览器发布目录。';
  for (const finish of pending.values()) finish({ error: failure });
};

function request(message, signal) {
  if (failure) return Promise.reject(new Error(failure));
  if (signal?.aborted) return Promise.reject(signal.reason);

  const id = ++sequence;
  return new Promise((resolve, reject) => {
    const abort = () => {
      pending.delete(id);
      reject(signal.reason);
    };
    pending.set(id, reply => {
      pending.delete(id);
      signal?.removeEventListener('abort', abort);
      if (reply.error) reject(new Error(reply.error));
      else resolve(reply.value);
    });
    signal?.addEventListener('abort', abort, { once: true });
    worker.postMessage({ id, ...message });
  });
}

export const ready = request({ mode: 'init' });

export async function analyze(mode, first, second, options, signal) {
  if (options.backend !== 'wasm') {
    if (!server) throw new Error('当前页面只提供浏览器 WASM 推理。');
    return analyzeServer(mode, first, second, options, signal);
  }
  await ready;
  return request({ mode, first, second, options }, signal);
}
