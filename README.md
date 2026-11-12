# SimdFace

简体中文 · [English](README.en.md)

SimdFace 是面向 .NET 10 的本地人脸检测与 1:1 比对项目。它用 YuNet 定位人脸和五个关键点，用 SFace 对齐人脸并提取归一化的 128 维特征。你可以在同一个网页切换浏览器 WASM、后端 CPU 和后端 Vulkan GPU，也可以从命令行或 C# 直接调用。

## 功能

- **人脸检测**。输出置信度、边界框和五个关键点。网页支持拖拽、预览和叠加标注。
- **照片比对与特征导出**。比对两张单人照片，或者导出特征 JSON，稍后在同一模型和提取流程下直接比较向量。
- **摄像头活体核验**。通过随机转头、闭眼睁眼或张嘴合嘴指令，结合防翻拍评分，分别给出活体和参考照匹配结论。
- **多种执行方式**。C# CPU 后端使用 SIMD 和并行卷积，Vulkan 后端把模型算子交给显卡驱动，浏览器可通过官方 .NET WASM AOT 独立运行。
- **大图处理**。网页支持 JPEG、PNG、WebP，单张最多 64 MiB 和 6400 万像素。托管解码器应用 EXIF 方向，检测坐标映射回校正方向后的原图，SFace 从原图采样。

## NuGet 安装

在 .NET 10 项目中安装推理核心。[NuGet.org](https://www.nuget.org/packages/SimdFace/1.1.0) · [1.1.0 发布说明](docs/releases/v1.1.0.md)

```powershell
dotnet add package SimdFace --version 1.1.0
```

核心包接收连续的 RGB24 像素，包含人脸检测、特征提取和活体核验 API，以及中文 IntelliSense 文档。模型权重及各自的许可证见仓库 [models/](models/)。图像解码、CLI 和网页测试台可从源码构建。[C# 调用示例](src/SimdFace/README.md)

## 快速开始

安装 [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) 和 PowerShell 7。仓库已包含 YuNet 和 SFace 的固定权重，克隆后即可使用。以下脚本会校验 SHA-256，并在文件缺失或损坏时重新下载。

```powershell
pwsh -File ./scripts/Get-Models.ps1
dotnet restore --locked-mode
dotnet build SimdFace.slnx -c Release --no-restore
```

### 网页测试台

启动 ASP.NET Core 服务。默认监听所有网卡的 5088 端口，本机可打开 [http://127.0.0.1:5088/](http://127.0.0.1:5088/)，局域网设备可访问 `http://<服务主机的局域网 IP>:5088/`。

```powershell
dotnet run -c Release --no-build --project src/SimdFace.Web
```

网页的照片检测、比对和特征提取默认使用后端 GPU（Vulkan），也可在「推理设备」中切换到后端 CPU。GPU 路径使用项目内嵌的 SPIR-V，需要兼容的 Vulkan loader 和显卡驱动。解码、图像对齐和输出后处理仍在 CPU 执行。Vulkan 后端面向 64 位 Windows 和 Linux，本文的性能数据来自 Windows 实测。

### 在同一个页面启用浏览器 WASM

安装 `wasm-tools` 工作负载并发布 AOT 资源，再让 ASP.NET Core 同源提供该资源目录和后端 API。

```powershell
dotnet workload install wasm-tools
dotnet publish src/SimdFace.Browser -c Release -p:RunAOTCompilation=true
$simdFaceWasm = Join-Path $PWD 'src/SimdFace.Browser/bin/Release/net10.0/publish/wwwroot'
dotnet run --project src/SimdFace.Web -c Release -- --urls 'http://*:5089' --SimdFace:BrowserAssets "$simdFaceWasm"
```

打开 [统一测试台](http://127.0.0.1:5089/)。浏览器 WASM 在页面打开时初始化，约 37 MiB 的权重在第一次处理照片时加载。浏览器模式在 Web Worker 中处理照片和向量。选择后端时，页面把文件发给本机服务，在结果中分开显示解码、推理和请求耗时。向量比对可在浏览器或后端 CPU 上执行。

纯静态部署可以发布整个 `wwwroot`，包括 `_framework` 和 `models`。本机可用 Python 运行 `python scripts/serve-wasm.py`，用 HTTP 服务该目录。

### 摄像头核验

打开测试台顶部的「摄像头核验」，选择一张单人正面照，允许摄像头访问，按提示转一次头，再完成随机出现的闭眼睁眼或张嘴合嘴动作。页面分别显示 SFace 照片匹配和活体结果。核验会检查转头前后的人脸一致性，再融合两次正脸的 SFace 特征，与参考照片比对。融合分数达到 0.61 且两次单独比对各达到 0.54 时，照片比对通过。MiniFASNetV2 在合适的帧上做防翻拍评分，MediaPipe Face Mesh 在正脸采样时估计 468 个三维点。这里的 Z 坐标是从 RGB 估计的相对深度。浏览器只采集帧，模型推理在 .NET 后端执行。照片比对和检测可用 Vulkan，防翻拍和网格用 CPU。

其他 .NET 应用可以直接使用核心库的 [`FaceVerificationSession`](src/SimdFace/FaceVerificationSession.cs)。加载 YuNet、SFace、MiniFASNetV2 和 Face Mesh，用参考照片生成 `EmbedForVerification` 特征，为每次核验创建一个会话，再按采集顺序调用 `ProcessFrame(rgb, width, height)`。帧使用紧密排列的 RGB24 像素。返回值包含当前动作、照片相似度和独立的活体结论，应用负责采集、显示提示和控制会话时限。

#### C# 活体检测示例

把仓库 `models/` 中的四份 ONNX 文件放到程序工作目录的 `models/` 下。将参考照解码为 RGB24，通过 `frames` 按采集顺序和实际帧间隔传入摄像头画面。示例使用 .NET 10 控制台项目的默认隐式 using。

```csharp
using SimdFace;

var detector = new FaceDetector(Path.Combine("models", FaceDetector.ModelFileName));
var recognizer = new FaceRecognizer(Path.Combine("models", FaceRecognizer.ModelFileName));
var antiSpoof = new FaceAntiSpoof(Path.Combine("models", FaceAntiSpoof.ModelFileName));
var mesh = new FaceMesh(Path.Combine("models", FaceMesh.ModelFileName));

async Task<bool> VerifyAsync(RgbFrame photo, IAsyncEnumerable<RgbFrame> frames,
    Action<FaceVerificationResult> showPrompt, CancellationToken cancellationToken = default)
{
    var faces = detector.Detect(photo.Pixels, photo.Width, photo.Height);
    if (faces.Count != 1)
        throw new ArgumentException("参考照片应恰好包含一张脸。", nameof(photo));

    var reference = recognizer.EmbedForVerification(
        photo.Pixels, photo.Width, photo.Height, faces[0].Landmarks, cancellationToken);
    var session = new FaceVerificationSession(detector, recognizer, antiSpoof, mesh, reference);

    await foreach (var frame in frames.WithCancellation(cancellationToken))
    {
        var result = session.ProcessFrame(frame.Pixels, frame.Width, frame.Height, cancellationToken);
        showPrompt(result);
        if (result.IsFinished)
            return result.Liveness == FaceLivenessDecision.Passed &&
                   result.Identity == FaceIdentityDecision.Match;
    }

    return false;
}

public sealed record RgbFrame(byte[] Pixels, int Width, int Height);
```

`showPrompt` 可读取 `Stage`、`TurnDirection`、`ExpressionPrompt` 和 `Hint`，显示转头、闭眼、睁眼、张嘴或合嘴提示。调用方可用取消令牌限制核验时长。`VerifyAsync` 只有在活体通过且人脸匹配时才返回 `true`。

本机可打开 [摄像头核验](http://localhost:5088/verify.html)。手机与电脑连接同一局域网时，可用 Windows 上的 PowerShell 7.4+ 启动 HTTPS。将参数换成电脑在该网段的 IP。

```powershell
pwsh -File ./scripts/Start-CameraLab.ps1 -LanAddress 192.168.1.3
```

脚本打印 HTTPS 地址和本地 CA 公钥证书下载地址。首次测试时在手机上安装并信任该证书，再打开 HTTPS 页面。iPhone 还需要在「设置 → 通用 → 关于本机 → 证书信任设置」启用完全信任。[Apple 证书信任说明](https://support.apple.com/zh-cn/102390)

仓库包含两份摄像头模型及其上游许可证。MiniFASNetV2 的全连接节点经过等价转换，Face Mesh 的常量节点在离线阶段折叠。准备脚本会校验源文件并对照转换前后输出。[MiniFASNetV2](https://github.com/yakhyo/face-anti-spoofing/releases/tag/weights) · [Face Mesh](https://github.com/yakhyo/mediapipe-face-mesh-onnx/releases/tag/weights) · [模型转换脚本](scripts/prepare-camera-models.py)

### 命令行

```powershell
dotnet run -c Release --no-build --project src/SimdFace.Cli -- detect photo.jpg
dotnet run -c Release --no-build --project src/SimdFace.Cli -- compare first.jpg second.jpg
dotnet run -c Release --no-build --project src/SimdFace.Cli -- compare first.jpg second.jpg --backend vulkan
```

`detect` 返回图片宽高、检测框、五个关键点和置信度。无脸时 `faces` 为空数组。`compare` 要求两张图各检出一张脸，输出 `similarity`、`threshold` 和 `isMatch`。两条命令都支持 `--models`、`--score` 和 `--backend cpu|vulkan`，默认检测阈值为 `0.8`。比对可用 `--threshold` 指定阈值。输入错误写入标准错误并返回退出码 `2`。

## 比对流程与阈值

YuNet 检测输入的长边最多为 640 像素。SFace 将原图人脸对齐为 `112 × 112`，输出归一化特征。CLI 默认使用一次对齐裁剪，阈值 `0.363` 来自 SFace 的 LFW 基准。网页使用三种裁剪及其镜像共六次提取，合并后归一化。网页默认同人阈值为 `0.68`，低于 `0.363` 判为不同人，中间显示无法确认。调低同人阈值会同时影响同人通过率和误认率，部署时应按实际样本校准。 [OpenCV SFace 基准](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)

特征 JSON 带有模型和提取流程标识。网页导出的六裁剪特征使用 `sface-2021dec` 和 `align5-6crop-v1`，可在浏览器与后端 CPU 间互用。

### 在 C# 中调用

引用 [核心项目](src/SimdFace/SimdFace.csproj) 和 [图像解码项目](src/SimdFace.Imaging/SimdFace.Imaging.csproj)。已有连续 RGB24 像素时可直接调用核心库。

```csharp
using System.IO;
using System.Linq;
using SimdFace;
using SimdFace.Imaging;

var detector = new FaceDetector(Path.Combine("models", FaceDetector.ModelFileName));
var recognizer = new FaceRecognizer(Path.Combine("models", FaceRecognizer.ModelFileName));

float[] Embed(string path)
{
    var image = ImageDecoder.ReadFile(path);
    var face = detector.Detect(image.Rgb, image.Width, image.Height).Single();
    return recognizer.EmbedForVerification(image.Rgb, image.Width, image.Height, face.Landmarks);
}

float similarity = FaceRecognizer.CosineSimilarity(Embed("first.jpg"), Embed("second.jpg"));
```

### HTTP API

`POST /api/detect`、`/api/compare`、`/api/extract` 和 `/api/vectors` 接受 `multipart/form-data`，请求头为 `X-SimdFace-Request: 1`。单文件用 `first`，双文件用 `first` 和 `second`。照片接口支持查询参数 `backend=cpu|vulkan` 和 `score`，比对接口支持 `threshold`。双图比对逐张解码，复用同一块原图 RGB24 缓冲区。

## 性能实测

Windows、NVIDIA GeForce RTX 3060，使用约 2447 万和 1258 万像素的 JPEG 预热后交错测量 12 轮。表格为中位数，单位毫秒。单图检测使用第一张。推理包含检测、对齐、特征提取和后处理，请求还包含本机上传、解码和响应。

| 操作 | 后端 | 解码 | 推理 | 请求 |
| --- | --- | ---: | ---: | ---: |
| 单图检测 | CPU | 168.0 | 129.0 | 358.8 |
| 单图检测 | Vulkan | 144.5 | 60.5 | 269.0 |
| 双图比对 | CPU | 175.5 | 829.0 | 1034.5 |
| 双图比对 | Vulkan | 204.0 | 221.5 | 490.8 |

CPU 在卷积输出可复用时原地执行 ReLU。Vulkan 把每张脸的六个裁剪合并成一次提交。本轮改动后，CPU 双图请求从 1503.1 ms 降到 1034.5 ms，Vulkan 双图推理从 240.5 ms 降到 221.5 ms，请求时间从 463.4 ms 变为 490.8 ms。测试中的检测结果一致，各后端的比对分数保留四位小数均为 `0.6345`。

## 可选构建与评测

### NetWasm

独立的 `SimdFace.NetWasm` 项目使用 NetWasm SDK 0.11.3，复用 C# 推理内核。Web Worker 使用浏览器解码和校正 EXIF 方向，再分块传入原图 RGB24。模型在宿主侧进行 SHA-256 校验，NetWasm 客体的网络权限为 `denyAll`。两条浏览器路径共用界面。 [NetWasm 互操作文档](https://github.com/zion-sati/NetWasm/blob/main/docs/javascript-interop.md)

```powershell
dotnet publish src/SimdFace.NetWasm -c Release
python scripts/serve-wasm.py src/SimdFace.NetWasm/bin/Release/netwasm0.1/publish
```

使用 localhost 或 HTTPS，静态部署时发布整个 `publish` 目录。Edge 实测通过 EXIF 旋转 JPEG、PNG、WebP 和 `8000 × 8000` JPEG 特征提取。无损 PNG 和 WebP 样本在两套浏览器运行时产生逐位一致的 Float32 特征。JPEG 解码路径不同，可能让输入像素和分数产生小幅差异。

### 离线模型对照

[评测脚本](benchmarks/compare-face-models.py) 通过 ONNX Runtime CPU 对照 SFace 和 EdgeFace-XXS，共用 YuNet 检测及单次对齐裁剪。它用 AT&T 身份 1–20 的异人对在 0.1% 目标误认率下校准阈值，再测试身份 21–40，并可用 KinFaceW-I 另测亲属误认。 [AT&T 数据](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) · [KinFaceW-I](https://www.kinfacew.com/download.html) · [EdgeFace 权重](https://github.com/yakhyo/edgeface-onnx/releases/tag/weights)

```powershell
python -m pip install opencv-python-headless==4.13.0.92 numpy==2.4.4 onnxruntime==1.24.3
python benchmarks/compare-face-models.py --att-zip att_faces.zip --edgeface edgeface_xxs.onnx --kinface-zip KinFaceW-I.zip
```

| 模型 | AT&T 同人通过 | AT&T 异人误认 | KinFaceW-I 亲属误认 | 单张特征 P50 |
| --- | ---: | ---: | ---: | ---: |
| SFace | 900 / 900 | 0 / 19,000 | 24 / 533 | 10.2 ms |
| EdgeFace-XXS | 900 / 900 | 23 / 19,000 | 33 / 533 | 2.3 ms |

这组结果来自本机四线程、单裁剪评测。网页六裁剪流程使用独立的同人阈值。

## 项目结构

| 项目 | 职责 |
| --- | --- |
| [SimdFace](src/SimdFace) | 固定模型的 C# ONNX 解析、CPU SIMD、Vulkan、对齐及向量比对 |
| [SimdFace.Imaging](src/SimdFace.Imaging) | 纯托管 SimdCodecs 解码及 EXIF 方向校正 |
| [SimdFace.Cli](src/SimdFace.Cli) | 检测和比对命令 |
| [SimdFace.Web](src/SimdFace.Web) | ASP.NET Core 测试台、HTTP API 和统一静态托管 |
| [SimdFace.Browser](src/SimdFace.Browser) | 官方 .NET WASM AOT 浏览器构建 |
| [SimdFace.NetWasm](src/SimdFace.NetWasm) | 独立的 NetWasm 浏览器构建 |

## 模型与许可证

运行时支持两份固定权重。CPU 推理所需的 ONNX 算子由 C# 实现，离线模型评测另外使用 Python、OpenCV 和 ONNX Runtime。

| 权重 | 用途 | 上游和许可证 |
| --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | 人脸检测和五点定位 | [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) · [MIT](models/YuNet.LICENSE.txt) |
| `face_recognition_sface_2021dec.onnx` | 128 维人脸特征 | [OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) · [Apache-2.0](models/SFace.LICENSE.txt) |

源码采用 [MIT 许可证](LICENSE)。图像解码使用 [SimdCodecs](https://github.com/sdcb/SimdCodecs)，借用的 Vulkan 绑定遵循 [SimdSkia 许可证](src/SimdFace/Vulkan/SimdSkia.LICENSE.txt)。相关项目包括 [SimdYolo](https://github.com/donet-org-cn/SimdYolo) 和 [SimdSkia](https://github.com/donet-org-cn/SimdSkia)。
