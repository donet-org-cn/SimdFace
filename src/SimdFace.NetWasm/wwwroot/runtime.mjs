import { open } from './browser/SimdFace.NetWasm.mjs';

const maxPixels = 64_000_000;
const chunkBytes = 512 * 1024;
const featureModel = 'sface-2021dec';
const featurePipeline = 'align5-6crop-v1';
const landmarkNames = ['rightEye', 'leftEye', 'nose', 'rightMouth', 'leftMouth'];

async function loadModel(api, name, recognition) {
  const response = await fetch(new URL('models/' + name, import.meta.url));
  if (!response.ok) throw new Error('模型加载失败，请先运行 Get-Models.ps1 并重新发布。');
  const bytes = new Uint8Array(await response.arrayBuffer());
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  const hash = Array.from(digest, value => value.toString(16).padStart(2, '0')).join('');
  if (hash !== api.ModelHash(recognition)) throw new Error('模型校验失败：' + name);
  api.BeginBuffer(bytes.length);
  for (let offset = 0; offset < bytes.length; offset += chunkBytes)
    api.AppendBytes(bytes.subarray(offset, offset + chunkBytes));
  api.LoadModel(recognition);
}

async function decodePhoto(api, file) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, {
      imageOrientation: 'from-image', colorSpaceConversion: 'none', premultiplyAlpha: 'none'
    });
  } catch {
    throw new Error('无法解码照片，请选择有效的 JPG、PNG 或 WebP。');
  }

  try {
    const { width, height } = bitmap;
    if (width <= 0 || height <= 0 || width * height > maxPixels)
      throw new Error('单张照片的像素数上限为 6400 万。');

    const tileWidth = Math.min(width, 8192);
    const tileHeight = tileWidth < width ? 1 : Math.min(height, Math.max(1, Math.floor(chunkBytes / (tileWidth * 4))));
    const canvas = new OffscreenCanvas(tileWidth, tileHeight);
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('浏览器无法创建图像解码缓冲区。');
    context.globalCompositeOperation = 'copy';
    context.imageSmoothingEnabled = false;
    const rgb = new Uint8Array(tileWidth * tileHeight * 3);
    api.BeginBuffer(width * height * 3);
    for (let y = 0; y < height; y += tileHeight) {
      const rows = Math.min(tileHeight, height - y);
      for (let x = 0; x < width; x += tileWidth) {
        const columns = Math.min(tileWidth, width - x);
        context.drawImage(bitmap, x, y, columns, rows, 0, 0, columns, rows);
        const rgba = context.getImageData(0, 0, columns, rows).data;
        let index = 0;
        for (let p = 0; p < rgba.length; p += 4) {
          rgb[index++] = rgba[p];
          rgb[index++] = rgba[p + 1];
          rgb[index++] = rgba[p + 2];
        }
        api.AppendBytes(rgb.subarray(0, index));
      }
    }
    return { width, height };
  } finally {
    bitmap.close();
  }
}

function readResult(data) {
  const count = data[0];
  const faces = [];
  let i = 1;
  for (let n = 0; n < count; n++) {
    const [x, y, width, height, score] = data.subarray(i, i + 5);
    i += 5;
    const landmarks = {};
    for (const name of landmarkNames) {
      landmarks[name] = { x: data[i++], y: data[i++] };
    }
    faces.push({ x, y, width, height, score, landmarks });
  }
  const feature = data.length - i === 128
    ? { model: featureModel, pipeline: featurePipeline, values: Array.from(data.subarray(i)) }
    : null;
  return { faces, feature };
}

function readVector(json) {
  const feature = JSON.parse(json);
  if (feature?.model !== featureModel || feature.pipeline !== featurePipeline ||
      !Array.isArray(feature.values) || feature.values.length !== 128 ||
      !feature.values.every(Number.isFinite)) {
    throw new Error('请选择由当前 SFace 提取流程导出的 128 维 JSON 特征。');
  }
  return Float32Array.from(feature.values);
}

export async function createEngine() {
  const api = await open();
  return {
    name: 'NetWasm',
    async loadModels() {
      await loadModel(api, 'face_detection_yunet_2023mar.onnx', false);
      await loadModel(api, 'face_recognition_sface_2021dec.onnx', true);
    },
    async analyzePhoto(file, score, extract) {
      const start = performance.now();
      const { width, height } = await decodePhoto(api, file);
      const result = readResult(api.AnalyzeImage(width, height, score, extract));
      return { width, height, elapsedMs: Math.round(performance.now() - start), ...result };
    },
    compareVectors: (first, second) => api.CompareVectors(readVector(first), readVector(second))
  };
}
