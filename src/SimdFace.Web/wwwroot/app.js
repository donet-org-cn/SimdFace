const byId = id => document.getElementById(id);
const uploads = byId('uploads');
const run = byId('run');
const score = byId('score');
const threshold = byId('threshold');
const backend = byId('backend');
const modes = [...document.querySelectorAll('[data-mode]')];
const maxFileSize = 16 * 1024 * 1024;
const maxPixels = 24_000_000;
let mode = 'detect';
let activeRequest = null;
const cards = ['照片 A', '照片 B'].map(makeCard);
setMode('detect');

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
      if (file.size > maxFileSize) throw new Error('图片大小超过 16 MiB。');
      let bitmap;
      try {
        bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
      } catch {
        throw new Error('无法预览此图片，请选择 JPG、PNG 或 WebP。');
      }
      try {
        if (current !== revision) return;
        if (bitmap.width * bitmap.height > maxPixels) throw new Error('图片像素数超过 2400 万。');
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
  modes.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.mode === mode)));
  uploads.classList.toggle('compare', compare);
  cards[1].card.hidden = !compare;
  byId('threshold-control').hidden = !compare;
  threshold.disabled = !compare;
  byId('mode-description').textContent = compare ? '每张照片各需一张人脸。检测和比对完成后直接给出判断。' : '查看单张照片中的所有检测结果。';
  resetResults();
}

function resetResults() {
  activeRequest?.abort();
  activeRequest = null;
  cards.forEach(({ overlay, info, file }) => {
    overlay.replaceChildren();
    info.textContent = file?.name ?? '';
  });
  const detail = mode === 'compare' ? '选择两张照片后，点击检测并比对。' : '选择照片后，点击开始检测。';
  showResult('等待测试', detail);
  refreshRun();
}

function refreshRun() {
  const compare = mode === 'compare';
  run.disabled = Boolean(activeRequest) || !cards[0].file || (compare && !cards[1].file);
  run.textContent = activeRequest ? '正在处理…' : compare ? '检测并比对 →' : '开始检测 →';
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
backend.addEventListener('change', resetResults);

byId('tester').addEventListener('submit', async event => {
  event.preventDefault();
  if (run.disabled) return;
  resetResults();
  const compare = mode === 'compare';
  const controller = new AbortController();
  activeRequest = controller;
  refreshRun();
  showResult('正在处理照片', compare ? '正在检测人脸并计算特征相似度。' : '正在检测人脸和五个关键点。');
  const data = new FormData();
  data.append('first', cards[0].file);
  if (compare) data.append('second', cards[1].file);
  const query = new URLSearchParams({ score: score.value, backend: backend.value });
  if (compare && threshold.value !== '') query.set('threshold', threshold.value);
  try {
    const response = await fetch(`/api/${mode}?${query}`, {
      method: 'POST', body: data, headers: { 'X-SimdFace-Request': '1' }, signal: controller.signal
    });
    const result = await response.json().catch(() => ({}));
    if (activeRequest !== controller) return;
    result.images?.forEach((image, i) => drawImageResult(image, cards[i]));
    const device = result.backend === 'vulkan' ? `Vulkan · ${result.deviceName}` : 'CPU';
    const runtime = `${device} · ${result.elapsedMs} ms`;
    if (compare && response.status === 422) {
      showResult('无法判断是否为同一人', `${result.message} ${runtime}`, 'uncertain');
      return;
    }
    if (!response.ok) throw new Error(result.message || `请求失败（HTTP ${response.status}）。`);
    if (compare) {
      const { similarity, threshold, differentThreshold, decision } = result.comparison;
      if (!decision) throw new Error('网页服务需要重启后再比对。');
      const details = `相似度 ${similarity.toFixed(4)} · 同人阈值 ${threshold} · 异人阈值 ${differentThreshold} · ${runtime}`;
      if (decision === 'uncertain') {
        showResult('无法确认是否为同一人', `${details}。分数处于两个阈值之间。`, 'uncertain');
      } else {
        const same = decision === 'same';
        showResult(same ? '模型判断为同一人' : '模型判断为不同人', details, same ? 'success' : 'different');
      }
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
