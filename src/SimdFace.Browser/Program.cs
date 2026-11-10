using System.Diagnostics;
using System.Runtime.InteropServices.JavaScript;
using System.Runtime.Versioning;
using System.Text.Json;
using System.Text.Json.Serialization;
using SimdFace.Imaging;

namespace SimdFace.Browser;

[SupportedOSPlatform("browser")]
public static partial class BrowserFace
{
    private static FaceDetector? detector;
    private static FaceRecognizer? recognizer;

    public static void Main() { }

    [JSExport]
    public static void LoadModels(byte[] detectorModel, byte[] recognizerModel)
    {
        var loadedDetector = new FaceDetector(detectorModel);
        var loadedRecognizer = new FaceRecognizer(recognizerModel);
        detector = loadedDetector;
        recognizer = loadedRecognizer;
    }

    [JSExport]
    public static string AnalyzePhoto(byte[] encodedImage, double scoreThreshold, bool extract)
    {
        if (detector is null || recognizer is null)
            throw new InvalidOperationException("请先加载模型。");
        if (encodedImage.Length > ImageDecoder.MaxFileBytes)
            throw new InvalidDataException("单张照片的文件上限为 64 MiB。");

        var timer = Stopwatch.StartNew();
        var image = ImageDecoder.Decode(encodedImage, "照片");
        var faces = detector.Detect(image.Rgb, image.Width, image.Height, (float)scoreThreshold);
        FaceFeature? feature = null;
        if (extract && faces.Count == 1)
        {
            float[] values = recognizer.EmbedForVerification(image.Rgb, image.Width, image.Height, faces[0].Landmarks);
            feature = new FaceFeature(FaceRecognizer.ModelId, FaceRecognizer.VerificationPipelineId, values);
        }

        var result = new PhotoResult(image.Width, image.Height, faces, timer.ElapsedMilliseconds, feature);
        return JsonSerializer.Serialize(result, BrowserJson.Default.PhotoResult);
    }

    [JSExport]
    public static double CompareVectors(string first, string second)
    {
        var a = ParseFeature(first);
        var b = ParseFeature(second);
        return FaceRecognizer.CosineSimilarity(a.Values, b.Values);
    }

    private static FaceFeature ParseFeature(string json)
    {
        var feature = JsonSerializer.Deserialize(json, BrowserJson.Default.FaceFeature);
        if (feature?.Model != FaceRecognizer.ModelId || feature.Pipeline != FaceRecognizer.VerificationPipelineId ||
            feature.Values?.Length != FaceRecognizer.EmbeddingSize)
            throw new InvalidDataException("请选择由当前 SFace 提取流程导出的 128 维 JSON 特征。");
        return feature;
    }
}

internal sealed record FaceFeature(string Model, string Pipeline, float[] Values);
internal sealed record PhotoResult(int Width, int Height, IReadOnlyList<DetectedFace> Faces, long ElapsedMs, FaceFeature? Feature);

[JsonSourceGenerationOptions(JsonSerializerDefaults.Web)]
[JsonSerializable(typeof(PhotoResult))]
[JsonSerializable(typeof(FaceFeature))]
internal sealed partial class BrowserJson : JsonSerializerContext;
