# SimdFace

SimdFace is a .NET 10 library for local face detection, 1:1 face verification, and camera liveness checks. YuNet returns face boxes and five landmarks, and SFace extracts normalized 128-dimensional embeddings. The library uses C# SIMD on the CPU and can run inference through Vulkan on 64-bit Windows or Linux with a compatible driver.

```sh
dotnet add package SimdFace --version 1.1.0
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

### Camera liveness verification

Get the [camera model weights](https://github.com/donet-org-cn/SimdFace/tree/v1.1.0/models) and place all four ONNX files under `models/` in your working directory. Supply the decoded RGB24 reference photo and frames in capture order at the camera cadence. This standalone example uses the default implicit usings in a .NET 10 console project.

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
        throw new ArgumentException("The reference photo must contain exactly one face.", nameof(photo));

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

Read `Stage`, `TurnDirection`, `ExpressionPrompt`, and `Hint` in `showPrompt` to guide the subject. Set the attempt time limit with a cancellation token. Vulkan can run YuNet and SFace. The anti-spoof and face mesh models run on the CPU.

The package includes the inference library, embedded Vulkan shaders, and Chinese IntelliSense documentation. Download the model weights separately. The repository also provides image decoding, a CLI, and a web app. [English documentation](https://github.com/donet-org-cn/SimdFace/blob/main/README.en.md) · [中文文档](https://github.com/donet-org-cn/SimdFace/blob/main/README.md)

SimdFace is MIT-licensed. The package includes the MIT notices for SimdFace and the Vulkan bindings adapted from SimdSkia. Model weights have separate upstream licenses.
