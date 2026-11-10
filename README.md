# SimdFace

SimdFace 是一个面向 .NET 10 的本地人脸检测和比对项目。名称延续 `Simd` 加能力名的方式，`Face` 涵盖人脸检测、关键点、对齐和特征提取。

## 运行 0.1.0

安装 .NET 10 SDK，在项目根目录运行。已有模型时，下载脚本只校验文件。

```powershell
.\scripts\Get-Models.ps1
dotnet restore --locked-mode
dotnet build -c Release --no-restore
dotnet run -c Release --no-build --project src/SimdFace.Cli -- detect photo.jpg
dotnet run -c Release --no-build --project src/SimdFace.Cli -- compare photo-a.jpg photo-b.jpg
```

- `detect` 输出图片宽高、每张人脸的边界框、检测置信度及五个关键点。无脸时 `faces` 为空数组。
- `compare` 要求每张图片恰好检出一张脸，输出 `similarity`、`threshold` 和 `isMatch`。`similarity` 为 `[-1, 1]` 内的余弦分数。默认阈值 `0.363` 来自 SFace 的 LFW 参考值，部署时需要按业务样本校准。 [OpenCV 模型说明](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)
- 两条命令都接受 `--models <目录>` 和 `--score <阈值>`。默认从当前工作目录的 `models` 加载权重，最低检测置信度为 `0.8`。比对可用 `--threshold <阈值>` 覆盖默认值。
- CLI 通过标准输出返回 JSON。文件、参数或比对的人脸数量有误时，通过标准错误输出原因并返回退出码 `2`。

CLI 通过 SimdCodecs 解码照片并应用 EXIF 方向，支持已注册的 JPEG、PNG、WebP 等格式。检测把长边限制在 640 像素以内，再将坐标映射回解码后的原尺寸。SFace 在原尺寸图像上采样五点对齐后的 `112 × 112` 人脸，生成归一化的 128 维向量。 [SimdCodecs](https://github.com/sdcb/SimdCodecs)

## 网页测试台

从项目根目录启动 ASP.NET Core 网页版，再打开 [本机测试台](http://127.0.0.1:5088/)。

```powershell
.\scripts\Get-Models.ps1
dotnet run -c Release --project src/SimdFace.Web
```

- **单图检测**在照片上叠加检测框和五个关键点，同时显示人脸数量与置信度。
- **推理设备**可在 CPU 和 Vulkan 间切换。同一批照片可以直接重跑对照，结果会显示本次使用的设备和推理用时。
- **双图比对**点击「检测并比对」后直接给出模型判断。每张照片需要恰好检出一张脸。默认相似度达到 `0.68` 时判断为同一人，低于 `0.363` 时判断为不同人，中间显示无法确认。同人阈值可调整。调至 `0.363` 或更低会取消无法确认区间。
- 网页对每张脸使用原始对齐裁剪、两个相近缩放裁剪，以及三者的左右镜像。六次 SFace 推理结果相加后归一化，仍得到 128 维向量。应用内可通过 `FaceRecognizer.EmbedForVerification` 使用同一流程。比对耗时会随推理次数增加。
- `0.68` 是网页为降低亲属误匹配设置的保守默认值。提高同人阈值也会让更多真实同人照片进入无法确认区间。余弦相似度是模型分数，长相相似的人仍可能获得高分。 [SFace 的基准阈值](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)
- 可选择或拖入 JPG、PNG、WebP，每张最多 16 MiB 和 2400 万像素。检测和比对在本机 C# 服务中运行，照片只在内存中处理。
- 如果端口被占用，在启动命令末尾追加 `-- --urls http://127.0.0.1:5089` 并打开对应地址。

在本机用同一套 C# 检测和对齐流程对照单次提取与增强提取。AT&T 的 400 张照片按身份分成四组，组内得到 1,800 对同人和 18,000 对异人。统一采用 `0.68` 时，明确判对的同人从 1,601 对增加到 1,649 对，两版在异人组均为 0 对误匹配。旧网页用 `0.60` 明确判对 1,711 对同人。 [AT&T Database of Faces](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)

KinFaceW-I 的 533 对父母与子女照片中有 532 对通过单脸检测。旧网页将 2 对判为同一人，增强提取配合 `0.68` 后为 0 对。其中 116 对母子照片在选定阈值后单独复核。该结果覆盖本次数据中的父母与子女。 [KinFaceW-I](https://www.kinfacew.com/datasets.html)

## Vulkan 加速

Vulkan 把 YuNet 和 SFace 的计算图提交给系统显卡驱动，输入图像的解码、对齐和输出后处理仍在 CPU 上完成。权重和中间张量保存在显存中。运行使用项目内嵌的 SPIR-V，不需要安装着色器编译器。当前 Vulkan 后端面向 64 位 Windows 和 Linux，本次实测在 Windows 完成。命令行可在 `detect` 或 `compare` 后添加 `--backend vulkan`。

本机 GTX 1050 Ti（驱动 560.94）在预热后，对每种工作负载、每个后端分别运行 24 次。检测使用两张公开样图和其中一张的放大版本，比对包括同人、异人和不同尺寸的照片。下表使用网页返回的 `elapsedMs`，包含检测、对齐、模型计算与后处理，计时从图片解码完成后开始。

| 网页操作 | CPU P50 / P95 | Vulkan P50 / P95 | P50 加速比 |
| --- | ---: | ---: | ---: |
| 单图检测 | 37 / 69 ms | 9 / 24 ms | 4.1 倍 |
| 双图增强比对 | 713 / 794 ms | 121.5 / 139 ms | 5.9 倍 |

本次样图中，CPU 和 Vulkan 的人脸数与比对结论一致，框和关键点的最大坐标差为 `0.00014` 像素，余弦相似度最大差为 `0.0000043`。

修改 GLSL 时，可以用 Khronos glslang 重新生成两个着色器。 [Khronos glslang](https://github.com/KhronosGroup/glslang)

```powershell
foreach ($name in 'tensor', 'pointwise') {
    glslang -V --target-env vulkan1.0 -Os -o "src/SimdFace/Vulkan/Shaders/$name.spv" "src/SimdFace/Vulkan/Shaders/$name.comp"
}
```

## 在 C# 中调用

核心库只依赖 .NET，可接收调用方提供的连续 RGB24 像素。模型实例可复用。

```csharp
using SimdFace;

var detector = new FaceDetector(Path.Combine("models", FaceDetector.ModelFileName));
var recognizer = new FaceRecognizer(Path.Combine("models", FaceRecognizer.ModelFileName));

var faceA = detector.Detect(rgbA, widthA, heightA).Single();
var faceB = detector.Detect(rgbB, widthB, heightB).Single();
float[] embeddingA = recognizer.Embed(rgbA, widthA, heightA, faceA.Landmarks);
float[] embeddingB = recognizer.Embed(rgbB, widthB, heightB, faceB.Landmarks);
float similarity = FaceRecognizer.CosineSimilarity(embeddingA, embeddingB);
```

## 模型与范围

首版支持 CPU 单图检测和本地 1:1 比对。卷积及向量点积使用 `System.Numerics.Vector<float>`，并提供标量路径。ONNX 解析和模型推理均由 C# 实现，支持下列两份固定权重，加载前校验 SHA-256。运行需要 .NET 和模型文件，无需安装 Python、OpenCV 或 ONNX Runtime。

| 权重 | 用途 | 上游来源与许可证 |
| --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | 人脸框与五点定位 | [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) · [MIT](models/YuNet.LICENSE.txt) |
| `face_recognition_sface_2021dec.onnx` | 128 维人脸向量 | [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) · [Apache-2.0](models/SFace.LICENSE.txt) |

`SimdFace` 是核心库，`SimdFace.Cli` 和 `SimdFace.Web` 共用 `SimdFace.Imaging` 的图片解码。SimdYolo 的纯托管推理是架构参考，SimdSkia 可作为桌面渲染层。 [SimdYolo](https://github.com/hnlyf168/SimdYolo) · [SimdSkia](https://github.com/donet-org-cn/SimdSkia)

SimdFace 源码采用 [MIT 许可证](LICENSE)。下载的模型权重及借用的 Vulkan 绑定适用各自的许可证。 [Vulkan 绑定许可证](src/SimdFace/Vulkan/SimdSkia.LICENSE.txt)
