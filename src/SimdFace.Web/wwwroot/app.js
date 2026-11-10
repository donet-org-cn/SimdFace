import { analyze, local, ready, server } from './backend.js';

const byId = id => document.getElementById(id);
const uploads = byId('uploads');
const run = byId('run');
const score = byId('score');
const threshold = byId('threshold');
const backend = byId('backend');
const modes = [...document.querySelectorAll('[data-mode]')];
const maxFileSize = 64 * 1024 * 1024;
const maxPixels = 64_000_000;
let mode = 'detect';
let activeRequest = null;
let runtimeState = local ? 'loading' : 'ready';
const featureA = byId('feature-a');
const featureB = byId('feature-b');
const featureJson = byId('feature-json');
const cards = ['照片 A', '照片 B'].map(makeCard);
const wasmOption = backend.querySelector('[value="wasm"]');
const gpuOption = backend.querySelector('[value="vulkan"]');
wasmOption.hidden = wasmOption.disabled = !local;
byId('backend-control').hidden = !server;
backend.value = local ? 'wasm' : 'cpu';
document.querySelectorAll('[data-local-only]').forEach(element => element.hidden = false);
setMode('detect');

if (local) {
  byId('engine-label').textContent = 'WASM AOT';
  const status = byId('wasm-status');
  status.hidden = false;
  status.textContent = '正在初始化 WASM…';
  ready.then(engine => {
    byId('engine-label').textContent = engine;
    runtimeState = 'ready';
    status.textContent = 'WASM 已就绪';
    refreshRun();
  }, error => {
    runtimeState = 'failed';
    status.textContent = error.message;
    refreshRun();
  });
}

function makeCard(title) {
  const card = byId('upload-template').content.firstElementChild.cloneNode(true);
  const input = card.querySelector('input');
  const dropzone = card.querySelector('.dropzone');
  const preview = card.querySelector('.preview');
  const canvas = card.querySelector('canvas');
  const empty = card.querySelector('.empty-upload');
  const info = card.querySelector('.image-info');
  const error = card.querySelector('.upload-error');
  const overlay = card.querySelector('.overlay');
  const state = { card, overlay, info, file: null };
  let revision = 0;
  card.querySelector('h3').textContent = title;
  input.setAttribute('aria-label', `选择${title}`);
  input.addEventListener('change', () => {
    const file = input.files[0];
    input.value = '';
    if (file) select(file);
  });
  dropzone.addEventListener('dragover', event => {
    event.preventDefault();
    dropzone.classList.add('dragging');
  });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragging'));
  dropzone.addEventListener('drop', event => {
    event.preventDefault();
    dropzone.classList.remove('dragging');
    if (event.dataTransfer.files.length) select(event.dataTransfer.files[0]);
  });
  uploads.append(card);
  return state;

  async function select(file) {
    const current = ++revision;
    state.file = null;
    resetResults();
    error.hidden = true;
    preview.hidden = true;
    empty.hidden = false;
    dropzone.classList.remove('has-image');
    info.textContent = '';
    try {
      if (file.size > maxFileSize) throw new Error('图片大小超过 64 MiB。');
      let bitmap;
      try {
        bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
      } catch {
        throw new Error('无法预览此图片，请选择 JPG、PNG 或 WebP。');
      }
      try {
        if (current !== revision) return;
        if (bitmap.width * bitmap.height > maxPixels) throw new Error('图片像素数超过 6400 万。');
        const scale = Math.min(1, 1200 / Math.max(bitmap.width, bitmap.height));
        canvas.width = Math.max(1, Math.round(bitmap.width * scale));
        canvas.height = Math.max(1, Math.round(bitmap.height * scale));
        canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        state.file = file;
        info.textContent = file.name;
        info.title = file.name;
        preview.hidden = false;
        empty.hidden = true;
        dropzone.classList.add('has-image');
      } finally {
        bitmap.close();
      }
    } catch (failure) {
      if (current !== revision) return;
      error.textContent = failure.message;
      error.hidden = false;
    }
    refreshRun();
  }
}

function setMode(next) {
  mode = next;
  const compare = mode === 'compare';
  const vectors = mode === 'vectors';
  gpuOption.disabled = vectors;
  if (vectors && backend.value === 'vulkan') backend.value = 'cpu';
  modes.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.mode === mode)));
  uploads.classList.toggle('compare', compare);
  uploads.hidden = vectors;
  byId('feature-files').hidden = !vectors;
  cards[1].card.hidden = !compare;
  byId('threshold-control').hidden = threshold.disabled = !(compare || vectors);
  byId('score-control').hidden = score.disabled = vectors;
  byId('mode-description').textContent = {
    detect: '查看单张照片中的所有检测结果。',
    compare: '每张照片各需一张人脸。检测和比对完成后直接给出判断。',
    extract: '从一张照片提取 128 维 SFace 特征。',
    vectors: '在浏览器或后端 CPU 比较两个特征 JSON 文件。'
  }[mode];
  resetResults();
}

function resetResults() {
  activeRequest?.abort();
  activeRequest = null;
  cards.forEach(({ overlay, info, file }) => {
    overlay.replaceChildren();
    info.textContent = file?.name ?? '';
  });
  byId('feature-export').hidden = true;
  featureJson.value = '';
  const detail = {
    detect: '选择照片后，点击开始检测。',
    compare: '选择两张照片后，点击检测并比对。',
    extract: '选择一张照片后，点击提取特征。',
    vectors: '选择两个特征 JSON 后，点击比对向量。'
  }[mode];
  showResult('等待测试', detail);
  refreshRun();
}

function refreshRun() {
  const inputsReady = mode === 'vectors' ? featureA.files.length && featureB.files.length
    : cards[0].file && (mode !== 'compare' || cards[1].file);
  const inBrowser = backend.value === 'wasm';
  byId('processing-note').textContent = inBrowser
    ? '照片与特征仅在浏览器中处理'
    : '照片与特征发送到所选后端处理';
  run.disabled = Boolean(activeRequest) || (inBrowser && runtimeState !== 'ready') || !inputsReady;
  run.textContent = activeRequest ? '正在处理…' : {
    detect: '开始检测 →', compare: '检测并比对 →', extract: '提取特征 →', vectors: '比对向量 →'
  }[mode];
}

function showResult(title, detail, state = 'idle') {
  byId('results').dataset.state = state;
  byId('result-title').textContent = title;
  byId('result-detail').textContent = detail;
  const marks = { success: '✓', different: '≠', uncertain: '?', error: '!' };
  byId('result-mark').textContent = marks[state] ?? '◎';
}

function drawImageResult({ width, height, faces }, { overlay, info }) {
  overlay.setAttribute('viewBox', `0 0 ${width} ${height}`);
  overlay.setAttribute('preserveAspectRatio', 'none');
  const radius = Math.max(width, height) / 185;
  overlay.replaceChildren(...faces.flatMap(face => [
    svgShape('rect', { x: face.x, y: face.y, width: face.width, height: face.height, class: 'face-box' }),
    ...Object.values(face.landmarks).map(point =>
      svgShape('circle', { cx: point.x, cy: point.y, r: radius, class: 'face-point' }))
  ]));
  const confidence = faces.length ? ` · 最高置信度 ${(faces[0].score * 100).toFixed(1)}%` : '';
  info.textContent = `${width} × ${height} · ${faces.length} 张人脸${confidence}`;
}

function svgShape(name, attributes) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
}

modes.forEach(button => button.addEventListener('click', () => {
  if (mode !== button.dataset.mode) setMode(button.dataset.mode);
}));
[score, threshold].forEach(input => input.addEventListener('input', resetResults));
[backend, featureA, featureB].forEach(input => input.addEventListener('change', resetResults));
byId('download-feature').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([featureJson.value], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'simdface-sface.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

byId('tester').addEventListener('submit', async event => {
  event.preventDefault();
  if (run.disabled) return;
  resetResults();
  const compare = mode === 'compare';
  const vectors = mode === 'vectors';
  const controller = new AbortController();
  activeRequest = controller;
  refreshRun();
  showResult(vectors ? '正在比较向量' : '正在处理照片',
    compare || vectors ? '正在计算特征相似度。' : '正在检测人脸和五个关键点。');
  const first = vectors ? featureA.files[0] : cards[0].file;
  const second = vectors ? featureB.files[0] : compare ? cards[1].file : undefined;
  const options = { score: score.value, backend: backend.value, threshold: threshold.value };
  const began = performance.now();
  try {
    const { result, ok, status } = await analyze(mode, first, second, options, controller.signal);
    if (activeRequest !== controller) return;
    result.images?.forEach((image, i) => drawImageResult(image, cards[i]));
    const device = result.backend === 'wasm' ? result.deviceName ?? 'WASM'
      : result.backend === 'vulkan' ? '后端 GPU · ' + result.deviceName : '后端 CPU';
    const runtime = Number.isFinite(result.decodeMs)
      ? device + ' · 解码 ' + result.decodeMs + ' ms · 推理 ' + result.elapsedMs + ' ms · 请求 ' + Math.round(performance.now() - began) + ' ms'
      : device + ' · ' + result.elapsedMs + ' ms';
    if (compare && status === 422) {
      showResult('无法判断是否为同一人', `${result.message} ${runtime}`, 'uncertain');
      return;
    }
    if (!ok) throw new Error(result.message || `请求失败（HTTP ${status}）。`);
    if (compare || vectors) {
      const { similarity, threshold, differentThreshold, decision } = result.comparison;
      if (!decision) throw new Error('网页服务需要重启后再比对。');
      const details = `相似度 ${similarity.toFixed(4)} · 同人阈值 ${threshold} · 异人阈值 ${differentThreshold} · ${runtime}`;
      if (decision === 'uncertain') {
        showResult('无法确认是否为同一人', `${details}。分数处于两个阈值之间。`, 'uncertain');
      } else {
        const same = decision === 'same';
        showResult(same ? '模型判断为同一人' : '模型判断为不同人', details, same ? 'success' : 'different');
      }
    } else if (mode === 'extract') {
      featureJson.value = JSON.stringify(result.feature, null, 2);
      byId('feature-export').hidden = false;
      showResult('已提取 128 维特征', runtime, 'success');
    } else {
      const count = result.images[0].faces.length;
      showResult(count ? `检出 ${count} 张人脸` : '未检出人脸',
        `检测阈值 ${score.value} · ${runtime}`, 'success');
    }
  } catch (failure) {
    if (activeRequest === controller) showResult('测试未完成', failure.message, 'error');
  } finally {
    if (activeRequest === controller) {
      activeRequest = null;
      refreshRun();
    }
  }
});
