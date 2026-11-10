import { dotnet } from './_framework/dotnet.js';

async function modelBytes(name) {
  const response = await fetch(new URL('models/' + name, import.meta.url));
  if (!response.ok) throw new Error('模型加载失败，请先运行 Get-Models.ps1 并重新发布。');
  return new Uint8Array(await response.arrayBuffer());
}

export async function createEngine() {
  const { getAssemblyExports, getConfig } = await dotnet.create();
  const exports = await getAssemblyExports(getConfig().mainAssemblyName);
  const api = exports.SimdFace.Browser.BrowserFace;
  return {
    name: '.NET WASM',
    async loadModels() {
      const [detector, recognizer] = await Promise.all([
        modelBytes('face_detection_yunet_2023mar.onnx'), modelBytes('face_recognition_sface_2021dec.onnx')
      ]);
      api.LoadModels(detector, recognizer);
    },
    async analyzePhoto(file, score, extract) {
      return JSON.parse(api.AnalyzePhoto(new Uint8Array(await file.arrayBuffer()), score, extract));
    },
    compareVectors: (first, second) => api.CompareVectors(first, second)
  };
}
