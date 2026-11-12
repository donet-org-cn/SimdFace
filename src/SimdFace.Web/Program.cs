using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using Microsoft.AspNetCore.WebUtilities;
using Microsoft.Extensions.FileProviders;
using Microsoft.Net.Http.Headers;
using SimdFace;
using SimdFace.Imaging;

const int maxFileBytes = ImageDecoder.MaxFileBytes;
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
var camera = new CameraVerification(detector, recognizer, gpu, modelDirectory);

var app = builder.Build();
using var browserFiles = OpenBrowserAssets(builder.Configuration["SimdFace:BrowserAssets"], app.Environment.ContentRootPath);
var types = new Microsoft.AspNetCore.StaticFiles.FileExtensionContentTypeProvider();
types.Mappings[".cer"] = "application/x-x509-ca-cert";
var staticFiles = new StaticFileOptions { ContentTypeProvider = types };
if (browserFiles is null)
{
    app.UseDefaultFiles();
    app.UseStaticFiles(staticFiles);
}
else
{
    app.Map("/host-config.js", branch => branch.Run(async context =>
    {
        context.Response.ContentType = "text/javascript";
        context.Response.Headers.CacheControl = "no-store";
        await context.Response.WriteAsync("export const server = true;", context.RequestAborted);
    }));
    app.UseDefaultFiles(new DefaultFilesOptions { FileProvider = browserFiles });
    types.Mappings[".mjs"] = "text/javascript";
    types.Mappings[".wasm"] = "application/wasm";
    types.Mappings[".onnx"] = "application/octet-stream";
    app.UseStaticFiles(new StaticFileOptions { FileProvider = browserFiles, ContentTypeProvider = types });
    app.UseStaticFiles(staticFiles);
}
app.MapPost("/api/detect", (HttpRequest request) => Analyze(request, PhotoMode.Detect));
app.MapPost("/api/compare", (HttpRequest request) => Analyze(request, PhotoMode.Compare));
app.MapPost("/api/extract", (HttpRequest request) => Analyze(request, PhotoMode.Extract));
app.MapPost("/api/vectors", CompareVectors);
app.MapPost("/api/camera/start", (HttpRequest request) => CameraRequest(request, null));
app.MapPost("/api/camera/{id}/frame", (HttpRequest request, string id) => CameraRequest(request, id));
app.MapPost("/api/camera/{id}/cancel", async (HttpRequest request, string id) =>
{
    if (request.Headers["X-SimdFace-Request"] != "1")
        return Results.BadRequest(new { Message = "缺少请求标记。" });
    await inferenceSlot.WaitAsync(request.HttpContext.RequestAborted);
    try { return camera.Cancel(id); }
    finally { inferenceSlot.Release(); }
});
try
{
    app.Run();
}
finally
{
    if (gpu.IsValueCreated)
        gpu.Value.Device.Dispose();
}

async Task<IResult> CameraRequest(HttpRequest request, string? id)
{
    if (request.Headers["X-SimdFace-Request"] != "1")
        return Results.BadRequest(new { Message = "缺少请求标记。" });

    var address = request.HttpContext.Connection.RemoteIpAddress;
    if (address?.IsIPv4MappedToIPv6 == true)
        address = address.MapToIPv4();
    if (!request.IsHttps && (address is null || !System.Net.IPAddress.IsLoopback(address)))
        return Results.BadRequest(new { Message = "请通过 HTTPS 进行摄像头核验。" });
    request.HttpContext.Response.Headers.CacheControl = "no-store";

    bool start = id is null;
    string backend = request.Query.TryGetValue("backend", out var requested) ? requested.ToString() : "vulkan";
    if (start && backend is not ("cpu" or "vulkan"))
        return Results.BadRequest(new { Message = "请选择 CPU 或 Vulkan。" });

    int fileLimit = start ? maxFileBytes : CameraVerification.FrameBytes;
    var bodyLimit = request.HttpContext.Features.Get<Microsoft.AspNetCore.Http.Features.IHttpMaxRequestBodySizeFeature>();
    if (bodyLimit is { IsReadOnly: false })
        bodyLimit.MaxRequestBodySize = fileLimit + 8192;
    var cancellation = request.HttpContext.RequestAborted;
    try
    {
        var files = await ReadFiles(request, 1, fileLimit, cancellation);
        if (!await inferenceSlot.WaitAsync(0, cancellation))
            return Results.Json(new { Message = "正在处理上一帧，请稍后重试。" }, statusCode: 429);
        try
        {
            return start ? camera.Start(files[0].Span, backend, cancellation)
                : camera.Frame(id!, files[0].Span, cancellation);
        }
        finally
        {
            inferenceSlot.Release();
        }
    }
    catch (BadHttpRequestException error) when (error.StatusCode == 413)
    {
        return Results.StatusCode(413);
    }
    catch (Exception error) when (error is InvalidDataException or ArgumentException or NotSupportedException)
    {
        return Results.BadRequest(new { Message = error.Message });
    }
    catch (Exception error) when (error is IOException or InvalidOperationException)
    {
        app.Logger.LogError(error, "Camera verification failed.");
        return Results.Json(new { Message = "摄像头核验服务暂时不可用，请检查模型文件和推理设备。" }, statusCode: 503);
    }
}

async Task<IResult> Analyze(HttpRequest request, PhotoMode mode)
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
        bool compare = mode == PhotoMode.Compare;
        float score = ReadThreshold(request, "score", FaceDetector.DefaultScoreThreshold, 0);
        if (score == 0)
            throw new InvalidDataException("检测阈值必须大于 0。");
        float threshold = compare ? ReadThreshold(request, "threshold", defaultSamePersonThreshold, -1) : 0;
        var files = await ReadFiles(request, compare ? 2 : 1, maxFileBytes, cancellation);
        var images = new ImageUpload[files.Length];
        for (int i = 0; i < files.Length; i++)
            images[i] = InspectImage(files[i], i == 0 ? "照片 A" : "照片 B");

        var gpuModels = backend == "vulkan" ? gpu.Value : null;
        var activeDetector = gpuModels?.Detector ?? detector;
        var activeRecognizer = gpuModels?.Recognizer ?? recognizer;
        var timer = new Stopwatch();
        var decodeTimer = new Stopwatch();
        var detections = new ImageResult[images.Length];
        // Keep only one decoded photo, including during full-resolution face alignment.
        var rgbBuffer = new byte[images.Max(image => image.RgbLength)];
        var embeddings = new float[]?[images.Length];
        for (int i = 0; i < images.Length; i++)
        {
            cancellation.ThrowIfCancellationRequested();
            var image = images[i];
            var rgb = rgbBuffer.AsSpan(0, image.RgbLength);
            decodeTimer.Start();
            Decode(image, rgb, cancellation);
            decodeTimer.Stop();

            timer.Start();
            var faces = activeDetector.Detect(rgb, image.Width, image.Height, score);
            detections[i] = new ImageResult(image.Width, image.Height, faces);
            if (mode != PhotoMode.Detect && faces.Count == 1 && (i == 0 || embeddings[0] is not null))
                embeddings[i] = activeRecognizer.EmbedForVerification(rgb, image.Width, image.Height, faces[0].Landmarks, cancellation);
            timer.Stop();
        }

        string? message = null;
        FeatureVector? feature = null;
        ComparisonResult? comparison = null;
        timer.Start();
        if (compare)
        {
            if (embeddings[0] is { } first && embeddings[1] is { } second)
                comparison = Compare(first, second, threshold);
            else
                message = $"比对要求每张图恰好一张脸。照片 A 检出 {detections[0].Faces.Count} 张，照片 B 检出 {detections[1].Faces.Count} 张。";
        }
        else if (mode == PhotoMode.Extract)
        {
            if (embeddings[0] is { } values)
                feature = new FeatureVector(FaceRecognizer.ModelId, FaceRecognizer.VerificationPipelineId, values);
            else
                message = $"提取特征要求恰好一张脸，检出了 {detections[0].Faces.Count} 张。";
        }

        var result = new AnalysisResult(detections, timer.ElapsedMilliseconds, decodeTimer.ElapsedMilliseconds,
            backend, gpuModels?.Device.DeviceName, feature, comparison, message);
        return message is null ? Results.Ok(result) : Results.UnprocessableEntity(result);
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

static async Task<IResult> CompareVectors(HttpRequest request)
{
    if (request.Headers["X-SimdFace-Request"] != "1")
        return Results.BadRequest(new { Message = "缺少请求标记。" });

    try
    {
        float threshold = ReadThreshold(request, "threshold", defaultSamePersonThreshold, -1);
        var files = await ReadFiles(request, 2, 16 * 1024, request.HttpContext.RequestAborted);
        var timer = Stopwatch.StartNew();
        var comparison = Compare(ReadFeature(files[0].Span), ReadFeature(files[1].Span), threshold);
        return Results.Ok(new { Backend = "cpu", ElapsedMs = timer.ElapsedMilliseconds, Comparison = comparison });
    }
    catch (Exception error) when (error is InvalidDataException or ArgumentException or JsonException)
    {
        return Results.BadRequest(new { Message = error is JsonException ? "特征 JSON 格式无效。" : error.Message });
    }
}

static float[] ReadFeature(ReadOnlySpan<byte> data)
{
    var feature = JsonSerializer.Deserialize<FeatureVector>(data, JsonSerializerOptions.Web);
    if (feature?.Model != FaceRecognizer.ModelId || feature.Pipeline != FaceRecognizer.VerificationPipelineId ||
        feature.Values?.Length != FaceRecognizer.EmbeddingSize)
        throw new InvalidDataException("请选择由当前 SFace 提取流程导出的 128 维 JSON 特征。");
    return feature.Values;
}

static ComparisonResult Compare(float[] first, float[] second, float threshold)
{
    float similarity = FaceRecognizer.CosineSimilarity(first, second);
    float differentThreshold = Math.Min(FaceRecognizer.ReferenceThreshold, threshold);
    bool isMatch = similarity >= threshold;
    string decision = isMatch ? "same" : similarity < differentThreshold ? "different" : "uncertain";
    return new ComparisonResult(similarity, threshold, differentThreshold, isMatch, decision);
}

static PhysicalFileProvider? OpenBrowserAssets(string? directory, string contentRoot)
{
    if (string.IsNullOrWhiteSpace(directory))
        return null;
    string fullPath = Path.GetFullPath(directory, contentRoot);
    if (!File.Exists(Path.Combine(fullPath, "index.html")) ||
        !File.Exists(Path.Combine(fullPath, "_framework", "dotnet.js")))
        throw new InvalidOperationException($"SimdFace:BrowserAssets must point to the published .NET WASM wwwroot: {fullPath}");
    return new PhysicalFileProvider(fullPath);
}

static async Task<ReadOnlyMemory<byte>[]> ReadFiles(HttpRequest request, int count, int fileLimit, CancellationToken cancellation)
{
    if (!MediaTypeHeaderValue.TryParse(request.ContentType, out var contentType) ||
        !contentType.MediaType.Equals("multipart/form-data", StringComparison.OrdinalIgnoreCase))
        throw new InvalidDataException("请选择要上传的文件。");
    string? boundary = HeaderUtilities.RemoveQuotes(contentType.Boundary).Value;
    if (string.IsNullOrEmpty(boundary) || boundary.Length > 128)
        throw new InvalidDataException("文件上传格式无效。");

    var reader = new MultipartReader(boundary, request.Body) { BodyLengthLimit = fileLimit };
    var files = new ReadOnlyMemory<byte>[count];
    int index = 0;
    while (await reader.ReadNextSectionAsync(cancellation) is { } section)
    {
        if (index >= count || section.AsFileSection() is null)
            throw new InvalidDataException("上传的文件数量不正确。");
        using var buffer = new MemoryStream();
        try
        {
            await section.Body.CopyToAsync(buffer, cancellation);
        }
        catch (InvalidDataException error)
        {
            string limit = fileLimit >= 1024 * 1024 ? $"{fileLimit / (1024 * 1024)} MiB" : $"{fileLimit / 1024} KiB";
            throw new InvalidDataException($"无法读取文件 {index + 1}。单个文件上限为 {limit}。", error);
        }
        files[index++] = buffer.GetBuffer().AsMemory(0, (int)buffer.Length);
    }
    if (index != count)
        throw new InvalidDataException($"请选择 {count} 个文件。");
    return files;
}

static ImageUpload InspectImage(ReadOnlyMemory<byte> data, string label)
{
    try
    {
        var (width, height) = ImageDecoder.GetDimensions(data.Span, label);
        return new ImageUpload(data, width, height, label);
    }
    catch (InvalidDataException error)
    {
        throw new InvalidDataException(
            $"无法读取{label}。请选择有效图片，且像素数不超过 {ImageDecoder.DefaultMaxPixels / 10_000} 万。", error);
    }
}

static void Decode(ImageUpload image, Span<byte> rgb, CancellationToken cancellation)
{
    try
    {
        ImageDecoder.DecodeInto(image.Data.Span, rgb, image.Label, cancellationToken: cancellation);
    }
    catch (InvalidDataException error)
    {
        throw new InvalidDataException($"无法解码{image.Label}。请选择有效图片。", error);
    }
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

internal enum PhotoMode { Detect, Compare, Extract }

internal sealed record ImageUpload(ReadOnlyMemory<byte> Data, int Width, int Height, string Label)
{
    public int RgbLength => Width * Height * 3;
}

internal sealed record ImageResult(int Width, int Height, IReadOnlyList<DetectedFace> Faces);
internal sealed record FeatureVector(string Model, string Pipeline, float[] Values);
internal sealed record ComparisonResult(float Similarity, float Threshold, float DifferentThreshold, bool IsMatch, string Decision);
internal sealed record AnalysisResult(ImageResult[] Images, long ElapsedMs, long DecodeMs, string Backend,
    string? DeviceName, FeatureVector? Feature, ComparisonResult? Comparison, string? Message);
internal sealed record VulkanModels(VulkanInference Device, FaceDetector Detector, FaceRecognizer Recognizer);
