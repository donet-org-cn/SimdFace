# SimdFace

[简体中文](README.md) · English

SimdFace is a local face detection and 1:1 verification project for .NET 10. It uses YuNet to locate faces and five facial landmarks, then aligns each face and extracts a normalized 128-dimensional SFace embedding. Run the same workflow in a browser, on a server CPU or Vulkan GPU, from the command line, or directly in C#.

## Features

- **Face detection.** Get bounding boxes, confidence scores, and five landmarks. The web lab supports drag and drop, previews, and overlays.
- **Photo verification and embedding export.** Compare two photos with one face each, or export embeddings as JSON and compare them later using the same model and extraction pipeline.
- **Multiple execution paths.** The C# CPU backend uses SIMD and parallel convolution. The Vulkan backend runs model operators on the GPU. The browser can run independently through the official .NET WASM AOT runtime.
- **Large photos.** The web lab accepts JPEG, PNG, and WebP up to 64 MiB and 64 megapixels per photo. The managed decoder applies EXIF orientation. Detection coordinates are mapped to the oriented image, and SFace samples at full resolution.

## Quick start

Install the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) and PowerShell 7. From the repository root, download the pinned model weights and build. The download script verifies SHA-256 and reuses valid files.

```powershell
pwsh -File ./scripts/Get-Models.ps1
dotnet restore --locked-mode
dotnet build SimdFace.slnx -c Release --no-restore
```

### Web lab

Start the ASP.NET Core host and open [http://127.0.0.1:5088/](http://127.0.0.1:5088/). The interface uses Simplified Chinese. Choose the server CPU or Vulkan GPU from the inference device selector.

```powershell
dotnet run -c Release --no-build --project src/SimdFace.Web
```

The GPU path ships embedded SPIR-V and requires a compatible Vulkan loader and graphics driver. Decoding, face alignment, and post-processing run on the CPU. The backend targets 64-bit Windows and Linux. The performance figures on this page were measured on Windows.

### Enable browser WASM on the same page

Install the `wasm-tools` workload and publish the AOT assets. The ASP.NET Core host can serve the resulting directory and the API on the same origin.

```powershell
dotnet workload install wasm-tools
dotnet publish src/SimdFace.Browser -c Release -p:RunAOTCompilation=true
$simdFaceWasm = Join-Path $PWD 'src/SimdFace.Browser/bin/Release/net10.0/publish/wwwroot'
dotnet run --project src/SimdFace.Web -c Release -- --urls http://127.0.0.1:5089 --SimdFace:BrowserAssets "$simdFaceWasm"
```

Open the [unified web lab](http://127.0.0.1:5089/). Browser WASM initializes when the page opens and loads about 37 MiB of model weights on the first photo operation. Browser mode processes photos and embeddings in a Web Worker. Selecting a server backend sends files to the local service and displays separate decoding, inference, and request times. Embedding comparison runs in the browser or on the server CPU.

For static deployment, publish the entire `wwwroot` directory, including `_framework` and `models`. Run `python scripts/serve-wasm.py` to serve it over HTTP locally.

### Command line

```powershell
dotnet run -c Release --no-build --project src/SimdFace.Cli -- detect photo.jpg
dotnet run -c Release --no-build --project src/SimdFace.Cli -- compare first.jpg second.jpg
dotnet run -c Release --no-build --project src/SimdFace.Cli -- compare first.jpg second.jpg --backend vulkan
```

`detect` returns image dimensions, bounding boxes, five landmarks per face, and confidence scores. It returns an empty `faces` array when no face is detected. `compare` requires one detected face per photo and returns `similarity`, `threshold`, and `isMatch`. Both commands accept `--models`, `--score`, and `--backend cpu|vulkan`. The default detection threshold is `0.8`. Use `--threshold` to set the comparison threshold. Input errors go to stderr with exit code `2`.

## Verification pipeline and thresholds

YuNet uses an input with a long edge of at most 640 pixels. SFace aligns a `112 × 112` crop sampled from the original image and returns a normalized embedding. The CLI uses one crop and defaults to the SFace LFW reference threshold of `0.363`. The web lab aggregates three crops and their mirrors, then normalizes the result. Its default match threshold is `0.68`, scores below `0.363` are classified as different, and scores in between are marked uncertain. Threshold changes affect both true matches and false matches. Calibrate the threshold on representative photos before deployment. [OpenCV SFace benchmarks](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)

Feature JSON includes the model and pipeline IDs. Web exports use `sface-2021dec` and `align5-6crop-v1`. They can be compared in the browser or on the server CPU.

### Use from C#

Reference the [core project](src/SimdFace/SimdFace.csproj) and the [imaging project](src/SimdFace.Imaging/SimdFace.Imaging.csproj). The core project also accepts tightly packed RGB24 pixels directly.

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

`POST /api/detect`, `/api/compare`, `/api/extract`, and `/api/vectors` accept `multipart/form-data` with the `X-SimdFace-Request: 1` header. Use `first` for a single file, or `first` and `second` for two. Photo endpoints support the `backend=cpu|vulkan` and `score` query parameters. Comparison endpoints support `threshold`. Two-photo requests decode one image at a time and reuse a single RGB24 buffer.

## Measured performance

Windows with an NVIDIA GeForce RTX 3060, using approximately 24.47 MP and 12.58 MP JPEGs. Each path ran for 12 interleaved rounds after warm-up. The table reports median milliseconds. Detection uses the first photo. Inference covers detection, alignment, embedding extraction, and post-processing. Request time also includes local upload, decoding, and response.

| Operation | Backend | Decode | Inference | Request |
| --- | --- | ---: | ---: | ---: |
| Single-photo detection | CPU | 168.0 | 129.0 | 358.8 |
| Single-photo detection | Vulkan | 144.5 | 60.5 | 269.0 |
| Two-photo verification | CPU | 175.5 | 829.0 | 1034.5 |
| Two-photo verification | Vulkan | 204.0 | 221.5 | 490.8 |

The CPU path applies ReLU in place when the convolution output can be reused. The Vulkan path batches six crops per face into one submission. In this run, the CPU verification request dropped from 1503.1 ms to 1034.5 ms. Vulkan inference dropped from 240.5 ms to 221.5 ms, while its request time moved from 463.4 ms to 490.8 ms. Detection results and each backend's comparison scores matched its baseline.

## Optional builds and evaluation

### NetWasm

The standalone `SimdFace.NetWasm` project uses NetWasm SDK 0.11.3 and shares the C# inference kernels. Its Web Worker decodes photos with browser APIs, applies EXIF orientation, and transfers full-resolution RGB24 in chunks. The host checks model SHA-256, and the NetWasm guest uses the `denyAll` network policy. Both browser targets share the same UI. [NetWasm interop documentation](https://github.com/zion-sati/NetWasm/blob/main/docs/javascript-interop.md)

```powershell
dotnet publish src/SimdFace.NetWasm -c Release
python scripts/serve-wasm.py src/SimdFace.NetWasm/bin/Release/netwasm0.1/publish
```

Serve the entire `publish` directory on localhost or HTTPS. Edge validation covered EXIF-rotated JPEG, PNG, WebP, and embedding extraction from an `8000 × 8000` JPEG. Lossless PNG and WebP test samples produced bit-identical Float32 embeddings across the two browser runtimes. JPEG decoders can produce small pixel and score differences.

### Offline model comparison

The [evaluation script](benchmarks/compare-face-models.py) compares SFace and EdgeFace-XXS with ONNX Runtime on the CPU. Both models use the same YuNet detections and single aligned crops. It calibrates each threshold at a target 0.1% false match rate using impostor pairs from AT&T identities 1–20. It then tests identities 21–40 and can separately evaluate false matches on KinFaceW-I relatives. [AT&T dataset](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) · [KinFaceW-I](https://www.kinfacew.com/download.html) · [EdgeFace weights](https://github.com/yakhyo/edgeface-onnx/releases/tag/weights)

```powershell
python -m pip install opencv-python-headless==4.13.0.92 numpy==2.4.4 onnxruntime==1.24.3
python benchmarks/compare-face-models.py --att-zip att_faces.zip --edgeface edgeface_xxs.onnx --kinface-zip KinFaceW-I.zip
```

| Model | AT&T true matches | AT&T false matches | KinFaceW-I false matches | Single embedding P50 |
| --- | ---: | ---: | ---: | ---: |
| SFace | 900 / 900 | 0 / 19,000 | 24 / 533 | 10.2 ms |
| EdgeFace-XXS | 900 / 900 | 23 / 19,000 | 33 / 533 | 2.3 ms |

These results use four CPU threads and one crop per image. The six-crop web pipeline uses a separate match threshold.

## Repository layout

| Project | Role |
| --- | --- |
| [SimdFace](src/SimdFace) | C# ONNX parser for pinned models, CPU SIMD, Vulkan, alignment, and cosine comparison |
| [SimdFace.Imaging](src/SimdFace.Imaging) | Managed SimdCodecs decoding and EXIF orientation |
| [SimdFace.Cli](src/SimdFace.Cli) | Detection and comparison commands |
| [SimdFace.Web](src/SimdFace.Web) | ASP.NET Core web lab, HTTP API, and unified static hosting |
| [SimdFace.Browser](src/SimdFace.Browser) | Official .NET WASM AOT browser target |
| [SimdFace.NetWasm](src/SimdFace.NetWasm) | Standalone NetWasm browser target |

## Models and licenses

The runtime supports two pinned models. The required CPU inference operators are implemented in C#. The separate offline model comparison uses Python, OpenCV, and ONNX Runtime.

| Weights | Purpose | Upstream and license |
| --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | Face boxes and five landmarks | [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) · [MIT](models/YuNet.LICENSE.txt) |
| `face_recognition_sface_2021dec.onnx` | 128-dimensional face embeddings | [OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) · [Apache-2.0](models/SFace.LICENSE.txt) |

The source code is distributed under the [MIT license](LICENSE). Image decoding uses [SimdCodecs](https://github.com/sdcb/SimdCodecs). The borrowed Vulkan bindings follow the [SimdSkia license](src/SimdFace/Vulkan/SimdSkia.LICENSE.txt). Related projects include [SimdYolo](https://github.com/donet-org-cn/SimdYolo) and [SimdSkia](https://github.com/donet-org-cn/SimdSkia).
