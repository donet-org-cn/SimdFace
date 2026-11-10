# SimdFace

SimdFace is a .NET 10 library for local face detection and 1:1 face verification. YuNet returns face boxes and five landmarks, and SFace extracts normalized 128-dimensional embeddings. The library uses C# SIMD on the CPU and can run inference through Vulkan on 64-bit Windows or Linux with a compatible driver.

```sh
dotnet add package SimdFace --version 1.0.0
```

Download the pinned YuNet and SFace ONNX weights with `pwsh -File ./scripts/Get-Models.ps1` from the repository root. The script verifies SHA-256. Pass the downloaded model paths to the constructors and supply tightly packed RGB24 pixels in row order. [Model download instructions](https://github.com/donet-org-cn/SimdFace/blob/main/README.en.md#quick-start)

```csharp
using System;
using System.IO;
using SimdFace;

var detector = new FaceDetector(Path.Combine("models", FaceDetector.ModelFileName));
var recognizer = new FaceRecognizer(Path.Combine("models", FaceRecognizer.ModelFileName));

float[] Extract(ReadOnlySpan<byte> rgb, int width, int height)
{
    var faces = detector.Detect(rgb, width, height);
    if (faces.Count != 1)
        throw new InvalidOperationException("Expected exactly one face.");

    return recognizer.EmbedForVerification(rgb, width, height, faces[0].Landmarks);
}

// Compare two embeddings from the same extraction pipeline:
// float score = FaceRecognizer.CosineSimilarity(first, second);
```

The package contains the inference library and its embedded Vulkan shaders. Download model weights separately. The repository also provides image decoding, a CLI, and a web app. [English documentation](https://github.com/donet-org-cn/SimdFace/blob/main/README.en.md) · [中文文档](https://github.com/donet-org-cn/SimdFace/blob/main/README.md)

SimdFace is MIT-licensed. The package includes the MIT notices for SimdFace and the Vulkan bindings adapted from SimdSkia. Model weights have separate upstream licenses.
