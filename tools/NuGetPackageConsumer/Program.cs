using SimdFace;

if (args is not [_] and not [_, "--vulkan"] and not [_, "--expect-no-vulkan-hardware"])
    throw new ArgumentException("Usage: NuGetPackageConsumer <model-directory> [--vulkan | --expect-no-vulkan-hardware]");
string modelDirectory = args[0];

if (args is [_, "--expect-no-vulkan-hardware"])
{
    // 软件设备可以被 Vulkan loader 枚举，SimdFace 应将其排除在硬件加速路径之外。
    try
    {
        using var softwareDevice = new VulkanInference();
    }
    catch (NotSupportedException error) when (error.Message.StartsWith("No hardware Vulkan device", StringComparison.Ordinal))
    {
        Console.WriteLine("SimdFace package correctly rejected software-only Vulkan.");
        return;
    }

    throw new InvalidOperationException("Software-only Vulkan was accepted as a hardware device.");
}

using var vulkan = args.Length == 2 ? new VulkanInference() : null;
string ModelPath(string file) => Path.Combine(modelDirectory, file);
var detector = new FaceDetector(ModelPath(FaceDetector.ModelFileName), vulkan);
var recognizer = new FaceRecognizer(ModelPath(FaceRecognizer.ModelFileName), vulkan);
var antiSpoof = new FaceAntiSpoof(ModelPath(FaceAntiSpoof.ModelFileName));
var mesh = new FaceMesh(ModelPath(FaceMesh.ModelFileName));

// 使用固定的灰色画面和预设关键点，在不引入图像解码依赖的情况下执行四个模型。
const int width = 192, height = 192;
var rgb = new byte[width * height * 3];
Array.Fill(rgb, (byte)128);
var landmarks = new FaceLandmarks(new(65, 76), new(127, 76), new(96, 100), new(76, 130), new(116, 130));
var face = new DetectedFace(36, 30, 120, 132, 1, landmarks);
var reference = recognizer.EmbedForVerification(rgb, width, height, landmarks);
if (reference.Length != FaceRecognizer.EmbeddingSize || FaceRecognizer.CosineSimilarity(reference, reference) < 0.999f)
    throw new InvalidOperationException("SFace produced an invalid embedding.");

float score = antiSpoof.LiveScore(rgb, width, height, face);
if (!float.IsFinite(score) || score is < 0 or > 1)
    throw new InvalidOperationException("MiniFASNetV2 produced an invalid score.");
if (mesh.Estimate(rgb, width, height, face) is { } points &&
    (points.Length != 468 * 3 || points.Any(value => !float.IsFinite(value))))
    throw new InvalidOperationException("Face Mesh produced invalid coordinates.");

var session = new FaceVerificationSession(detector, recognizer, antiSpoof, mesh, reference);
var frame = session.ProcessFrame(rgb, width, height);
if (frame.IsFinished || frame.Stage != FaceVerificationStage.Frontal || frame.Hint != FaceVerificationHint.CenterFace)
    throw new InvalidOperationException("A blank frame must leave verification pending and request a visible face.");

Console.WriteLine($"SimdFace package smoke test passed on {vulkan?.DeviceName ?? "CPU"}.");
