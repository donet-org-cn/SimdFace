const byId = id => document.getElementById(id);
const form = byId('camera-form');
const video = byId('camera');
const cameraView = byId('camera-view');
const progress = byId('camera-progress');
const reference = byId('reference');
const backend = byId('camera-backend');
const start = byId('camera-start');
const stop = byId('camera-stop');
const overlay = byId('face-box');
const overlayContext = overlay.getContext('2d');
const capture = document.createElement('canvas');
const captureContext = capture.getContext('2d');
const frameInterval = 150;
const eyeFrameInterval = 90;
const maxFrameSide = 640;
const maxPhotoBytes = 64 * 1024 * 1024;
const headers = { 'X-SimdFace-Request': '1' };
const actionPrompts = {
  neutral: ['请重新正对镜头', '双眼自然睁开，嘴巴合上。'],
  closeEyes: ['请闭上双眼', '闭眼约半秒，然后睁开。'],
  openEyes: ['请睁开双眼', '继续正对镜头。'],
  openMouth: ['请张开嘴巴', '保持片刻，等下一条提示。'],
  closeMouth: ['请合上嘴巴', '继续正对镜头。']
};
let active = null;
let mesh = null;

const supported = window.isSecureContext && Boolean(navigator.mediaDevices?.getUserMedia);
start.disabled = !supported;
if (!supported)
  status('idle', '摄像头不可用', window.isSecureContext
    ? '当前浏览器没有提供摄像头接口。'
    : '请通过受信任的 HTTPS 或本机 localhost 打开。');

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (active || !supported || !reference.files.length) return;
  const photo = reference.files[0];
  if (photo.size > maxPhotoBytes) {
    status('failed', '参考照片过大', '单张参考照片上限为 64 MiB。');
    return;
  }

  const operation = { controller: new AbortController(), stream: null, session: null, timer: null, deadline: null, frameInterval };
  active = operation;
  start.disabled = reference.disabled = backend.disabled = true;
  stop.disabled = false;
  progress.value = 0;
  setText('identity-result', '待核验');
  setText('liveness-result', '待核验');
  setText('device-result', '—');
  setText('timing', '');
  byId('mesh-panel').hidden = true;
  mesh = null;
  status('running', '正在打开摄像头', '请允许浏览器访问摄像头。');

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: 'user' }, width: { ideal: 640 }, height: { ideal: 480 } }
    });
    if (active !== operation) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }
    operation.stream = stream;
    stream.getVideoTracks()[0].addEventListener('ended', () => {
      if (active === operation) {
        finish(operation);
        status('failed', '摄像头已断开', '请重新打开摄像头。');
      }
    });
    video.srcObject = stream;
    await video.play();
    if (active !== operation) return;
    const scale = Math.min(1, maxFrameSide / Math.max(video.videoWidth, video.videoHeight));
    capture.width = overlay.width = Math.max(1, Math.round(video.videoWidth * scale));
    capture.height = overlay.height = Math.max(1, Math.round(video.videoHeight * scale));
    cameraView.style.aspectRatio = capture.width + ' / ' + capture.height;
    byId('camera-placeholder').hidden = true;
    if (window.matchMedia('(max-width: 680px)').matches)
      cameraView.scrollIntoView({ block: 'nearest' });
    status('running', '正在分析参考照片', '请把脸移到画面中央。');
    const reply = await post('/api/camera/start?backend=' + backend.value, photo, photo.name, operation);
    if (active !== operation) return;
    operation.session = reply.session;
    operation.deadline = setTimeout(() => {
      if (active !== operation) return;
      finish(operation);
      status('failed', '核验超时', '请重新开始。');
    }, (reply.seconds + 5) * 1000);
    setText('device-result', reply.backend === 'vulkan' ? reply.deviceName + ' · Vulkan' : 'CPU SIMD');
    status('running', '请正对镜头', '让整张脸留在画面内。');
    await frame(operation);
  } catch (error) {
    fail(operation, error);
  }
});

stop.addEventListener('click', () => {
  if (!active) return;
  finish(active);
  status('idle', '核验已停止', '可以重新开始。');
});
window.addEventListener('pagehide', () => { if (active) finish(active); });
byId('mesh-angle').addEventListener('input', drawMesh);
byId('mesh-panel').addEventListener('toggle', drawMesh);

async function post(url, file, name, operation) {
  const data = new FormData();
  data.append('first', file, name);
  const response = await fetch(url, { method: 'POST', headers, body: data, signal: operation.controller.signal });
  if (response.status === 429 && operation.session) return null;
  const result = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(result.message || (response.status === 413 ? '上传文件过大。' : '核验请求失败。'));
  }
  return result;
}

async function frame(operation) {
  if (active !== operation) return;
  const began = performance.now();
  try {
    if (video.readyState >= 2) {
      captureContext.drawImage(video, 0, 0, capture.width, capture.height);
      const file = await new Promise(resolve => capture.toBlob(resolve, 'image/jpeg', 0.91));
      if (active !== operation) return;
      if (!file) throw new Error('无法读取摄像头画面。');
      const result = await post('/api/camera/' + operation.session + '/frame', file, 'frame.jpg', operation);
      if (active !== operation) return;
      if (result) {
        operation.frameInterval = result.action === 'closeEyes' || result.action === 'openEyes' ? eyeFrameInterval : frameInterval;
        showFrame(result, performance.now() - began);
        if (result.state !== 'running') {
          finish(operation, false);
          return;
        }
      }
    }
    if (active === operation) {
      const delay = Math.max(35, operation.frameInterval - (performance.now() - began));
      operation.timer = setTimeout(() => { void frame(operation); }, delay);
    }
  } catch (error) {
    fail(operation, error);
  }
}

function showFrame(result, latency) {
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  if (result.face) {
    const f = result.face;
    overlayContext.strokeStyle = '#5affad';
    overlayContext.lineWidth = 2;
    overlayContext.strokeRect(f.x, f.y, f.width, f.height);
  }
  const identities = { pending: '待核验', match: '匹配', different: '不匹配', uncertain: '无法确认' };
  const liveness = { pending: '检测中', passed: '通过', suspectedSpoof: '疑似翻拍', uncertain: '无法确认' };
  const score = result.similarity == null ? '' : ' · ' + result.similarity.toFixed(3);
  const threshold = result.matchThreshold == null ? '' : '（匹配线 ' + result.matchThreshold.toFixed(2) + '）';
  setText('identity-result', identities[result.identity] + score + threshold);
  setText('liveness-result', liveness[result.liveness] || '待核验');
  if (result.elapsedMs != null)
    setText('timing', '本帧服务处理 ' + result.elapsedMs + ' ms · 请求 ' + Math.round(latency) + ' ms');
  if (result.mesh?.length === 468 * 3) {
    mesh = result.mesh;
    byId('mesh-panel').hidden = false;
    drawMesh();
  }

  if (result.state === 'finished') {
    const passed = result.identity === 'match' && result.liveness === 'passed';
    const uncertain = result.identity === 'uncertain' && result.liveness === 'passed';
    progress.value = result.liveness === 'passed' ? 4 : result.step;
    const detail = result.message || (passed ? '参考照片匹配，活体检测通过。'
      : uncertain ? '活体通过，照片相似度不足以确认是同一人。请在光照均匀时正对镜头重试。'
      : '活体通过，镜头前的人与参考照片不匹配。');
    status(passed ? 'passed' : 'failed', passed ? '核验通过' : uncertain ? '身份待确认' : '核验未通过', detail);
    return;
  }
  if (result.state === 'timeout') {
    status('failed', '核验超时', result.message);
    return;
  }
  progress.value = result.step;
  const direction = result.direction > 0 ? '左' : '右';
  const instructions = [
    ['请正对镜头', '让整张脸留在画面内。'],
    ['请向自己的' + direction + '侧缓慢转头', '到位后停留片刻。'],
    actionPrompts[result.action] ?? actionPrompts.neutral,
    ['请正对镜头', '保持片刻，正在确认核验结果。']
  ];
  const [title, detail] = instructions[result.step];
  status('running', title, result.message || detail);
}

function drawMesh() {
  if (!mesh || !byId('mesh-panel').open) return;
  const canvas = byId('mesh');
  const ctx = canvas.getContext('2d');
  const points = [];
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < mesh.length; i += 3) {
    const point = mesh.slice(i, i + 3);
    points.push(point);
    cx += point[0]; cy += point[1]; cz += point[2];
  }
  cx /= points.length; cy /= points.length; cz /= points.length;
  const extent = Math.max(...points.map(([x, y]) => Math.hypot(x - cx, y - cy)));
  if (!Number.isFinite(extent) || extent < 1) return;
  const angle = Number(byId('mesh-angle').value) * Math.PI / 180;
  const cos = Math.cos(angle), sin = Math.sin(angle);
  const rotated = points.map(([x, y, z]) => [
    (x - cx) * cos + (z - cz) * sin, y - cy, (z - cz) * cos - (x - cx) * sin
  ]);
  const scale = 0.43 * canvas.height / extent;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const [x, y, z] of rotated.sort((a, b) => b[2] - a[2])) {
    ctx.fillStyle = 'hsl(174 78% ' + (62 - Math.max(-15, Math.min(20, z * scale / 3))) + '%)';
    ctx.beginPath();
    ctx.arc(canvas.width / 2 + x * scale, canvas.height / 2 + y * scale, 1.45, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function fail(operation, error) {
  if (active !== operation) return;
  finish(operation);
  const detail = {
    NotAllowedError: '请在浏览器中允许此站点访问摄像头。',
    NotFoundError: '未发现可用摄像头，请在带摄像头的设备上打开。',
    NotReadableError: '摄像头正在被其他应用使用，或当前无法访问。'
  }[error.name] || error.message;
  status('failed', '无法完成核验', detail);
}

function finish(operation, cancel = true) {
  if (active !== operation) return;
  active = null;
  clearTimeout(operation.timer);
  clearTimeout(operation.deadline);
  operation.controller.abort();
  operation.stream?.getTracks().forEach(track => track.stop());
  video.srcObject = null;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  byId('camera-placeholder').hidden = false;
  byId('camera-placeholder').textContent = '摄像头已关闭';
  start.disabled = reference.disabled = backend.disabled = false;
  stop.disabled = true;
  if (cancel && operation.session)
    fetch('/api/camera/' + operation.session + '/cancel', { method: 'POST', headers, keepalive: true }).catch(() => {});
}

function status(state, title, detail) {
  if (cameraView.dataset.state !== state) {
    cameraView.dataset.state = state;
    byId('camera-result').dataset.state = state;
  }
  progress.hidden = state === 'idle' || (state === 'failed' && progress.value === 0);
  setText('camera-status', title);
  setText('camera-instruction', detail);
}

function setText(id, text) {
  const element = byId(id);
  if (element.textContent !== text) element.textContent = text;
}
