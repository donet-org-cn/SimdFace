using System.Diagnostics;
using System.Globalization;
using Microsoft.AspNetCore.WebUtilities;
using Microsoft.Net.Http.Headers;
using SimdFace;
using SimdFace.Imaging;

const int maxFileBytes = 16 * 1024 * 1024;
const float defaultSamePersonThreshold = 0.68f;
var builder = WebApplication.CreateBuilder(args);
builder.WebHost.ConfigureKestrel(options => options.Limits.MaxRequestBodySize = 2 * maxFileBytes + 64 * 1024);
string modelDirectory = builder.Configuration["SimdFace:Models"] ??
    Path.GetFullPath(Path.Combine(builder.Environment.ContentRootPath, "..", "..", "models"));
var detector = new FaceDetector(Path.Combine(modelDirectory, FaceDetector.ModelFileName));
var recognizer = new FaceRecognizer(Path.Combine(modelDirectory, FaceRecognizer.ModelFileName));
var gpu = new Lazy<VulkanModels>(() =>
{
    var device = new VulkanInference();
    try
    {
        return new VulkanModels(device,
            new FaceDetector(Path.Combine(modelDirectory, FaceDetector.ModelFileName), device),
            new FaceRecognizer(Path.Combine(modelDirectory, FaceRecognizer.ModelFileName), device));
    }
    catch
    {
        device.Dispose();
        throw;
    }
});
using var inferenceSlot = new SemaphoreSlim(1);

var app = builder.Build();
app.UseDefaultFiles();
app.UseStaticFiles();
app.MapPost("/api/detect", (HttpRequest request) => Analyze(request, compare: false));
app.MapPost("/api/compare", (HttpRequest request) => Analyze(request, compare: true));
try
{
    app.Run();
}
finally
{
    if (gpu.IsValueCreated)
        gpu.Value.Device.Dispose();
}

async Task<IResult> Analyze(HttpRequest request, bool compare)
{
    if (request.Headers["X-SimdFace-Request"] != "1")
        return Results.BadRequest(new { Message = "缺少请求标记。" });

    string backend = request.Query.TryGetValue("backend", out var requested) ? requested.ToString() : "cpu";
    if (backend is not ("cpu" or "vulkan"))
        return Results.BadRequest(new { Message = "请选择 CPU 或 Vulkan。" });

    var cancellation = request.HttpContext.RequestAborted;
    if (!await inferenceSlot.WaitAsync(0, cancellation))
        return Results.Json(new { Message = "正在处理另一张图片，请稍后再试。" }, statusCode: 429);

    try
    {
        float score = ReadThreshold(request, "score", FaceDetector.DefaultScoreThreshold, 0);
        if (score == 0)
            throw new InvalidDataException("检测阈值必须大于 0。");
        float threshold = compare ? ReadThreshold(request, "threshold", defaultSamePersonThreshold, -1) : 0;
        var images = await ReadImages(request, compare ? 2 : 1, cancellation);
        var gpuModels = backend == "vulkan" ? gpu.Value : null;
        var activeDetector = gpuModels?.Detector ?? detector;
        var activeRecognizer = gpuModels?.Recognizer ?? recognizer;
        string? deviceName = gpuModels?.Device.DeviceName;
        var timer = Stopwatch.StartNew();
        var detections = new ImageResult[images.Count];
        for (int i = 0; i < images.Count; i++)
        {
            cancellation.ThrowIfCancellationRequested();
            var image = images[i];
            detections[i] = new ImageResult(image.Width, image.Height, activeDetector.Detect(image.Rgb, image.Width, image.Height, score));
        }

        if (!compare)
            return Results.Ok(new { Images = detections, ElapsedMs = timer.ElapsedMilliseconds, Backend = backend, DeviceName = deviceName });

        if (detections.Any(image => image.Faces.Count != 1))
            return Results.UnprocessableEntity(new
            {
                Message = $"比对要求每张图恰好一张脸。照片 A 检出 {detections[0].Faces.Count} 张，照片 B 检出 {detections[1].Faces.Count} 张。",
                Images = detections,
                ElapsedMs = timer.ElapsedMilliseconds,
                Backend = backend,
                DeviceName = deviceName
            });

        float[] Embed(int index)
        {
            cancellation.ThrowIfCancellationRequested();
            var image = images[index];
            return activeRecognizer.EmbedForVerification(image.Rgb, image.Width, image.Height, detections[index].Faces[0].Landmarks, cancellation);
        }

        float similarity = FaceRecognizer.CosineSimilarity(Embed(0), Embed(1));
        float differentThreshold = Math.Min(FaceRecognizer.ReferenceThreshold, threshold);
        bool isMatch = similarity >= threshold;
        string decision = isMatch ? "same" : similarity < differentThreshold ? "different" : "uncertain";
        return Results.Ok(new
        {
            Images = detections,
            ElapsedMs = timer.ElapsedMilliseconds,
            Backend = backend,
            DeviceName = deviceName,
            Comparison = new { Similarity = similarity, Threshold = threshold, DifferentThreshold = differentThreshold, IsMatch = isMatch, Decision = decision }
        });
    }
    catch (Exception error) when (backend == "vulkan" && error is (NotSupportedException or InvalidOperationException))
    {
        return Results.Json(new { Message = $"Vulkan 执行失败，请尝试切换到 CPU。{error.Message}" }, statusCode: 503);
    }
    catch (Exception error) when (error is InvalidDataException or ArgumentException or NotSupportedException)
    {
        return Results.BadRequest(new { error.Message });
    }
    finally
    {
        inferenceSlot.Release();
    }
}

static async Task<List<RgbImage>> ReadImages(HttpRequest request, int count, CancellationToken cancellation)
{
    if (!MediaTypeHeaderValue.TryParse(request.ContentType, out var contentType) ||
        !contentType.MediaType.Equals("multipart/form-data", StringComparison.OrdinalIgnoreCase))
        throw new InvalidDataException("请选择图片文件。");
    string? boundary = HeaderUtilities.RemoveQuotes(contentType.Boundary).Value;
    if (string.IsNullOrEmpty(boundary) || boundary.Length > 128)
        throw new InvalidDataException("图片上传格式无效。");

    var reader = new MultipartReader(boundary, request.Body) { BodyLengthLimit = maxFileBytes };
    var images = new List<RgbImage>(count);
    while (await reader.ReadNextSectionAsync(cancellation) is { } section)
    {
        if (images.Count >= count || section.AsFileSection() is null)
            throw new InvalidDataException("上传的图片数量不正确。");
        using var buffer = new MemoryStream();
        await section.Body.CopyToAsync(buffer, cancellation);
        var label = images.Count == 0 ? "照片 A" : "照片 B";
        try
        {
            images.Add(ImageDecoder.Decode(buffer.GetBuffer().AsSpan(0, (int)buffer.Length), label,
                maxPixels: 24_000_000, cancellationToken: cancellation));
        }
        catch (InvalidDataException)
        {
            throw new InvalidDataException($"无法解码{label}。请选择有效图片，且像素数不超过 2400 万。");
        }
    }
    if (images.Count != count)
        throw new InvalidDataException($"请选择 {count} 张图片。");
    return images;
}

static float ReadThreshold(HttpRequest request, string key, float defaultValue, float min)
{
    if (!request.Query.TryGetValue(key, out var input))
        return defaultValue;
    if (!float.TryParse(input.ToString(), NumberStyles.Float, CultureInfo.InvariantCulture, out float value) ||
        !float.IsFinite(value) || value < min || value > 1)
        throw new InvalidDataException("阈值超出范围，请检查输入。");
    return value;
}

internal sealed record ImageResult(int Width, int Height, IReadOnlyList<DetectedFace> Faces);
internal sealed record VulkanModels(VulkanInference Device, FaceDetector Detector, FaceRecognizer Recognizer);
