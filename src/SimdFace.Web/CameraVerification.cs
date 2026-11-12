using System.Buffers;
using System.Diagnostics;
using SimdFace;
using SimdFace.Imaging;

/// <summary>接收上传的参考照和摄像头画面，并接入核心库的核验会话。</summary>
/// <param name="cpuDetector">共享的 CPU 人脸检测器。</param>
/// <param name="cpuRecognizer">共享的 CPU 人脸特征提取器。</param>
/// <param name="vulkan">按需加载的 Vulkan 设备、人脸检测器和特征提取器。</param>
/// <param name="modelDirectory">防翻拍和人脸网格模型权重所在的目录。</param>
internal sealed class CameraVerification(
    FaceDetector cpuDetector,
    FaceRecognizer cpuRecognizer,
    Lazy<VulkanModels> vulkan,
    string modelDirectory)
{
    /// <summary>编码后的摄像头帧允许的最大字节数。</summary>
    public const int FrameBytes = 512 * 1024;
    private const int MaxFramePixels = 640 * 640;
    private const int SessionLimit = 16;
    private static readonly TimeSpan SessionLifetime = TimeSpan.FromSeconds(45);
    // 摄像头接口和照片接口共用推理信号量，由入口串行访问会话表。
    private readonly Dictionary<string, Session> sessions = new(StringComparer.Ordinal);
    private readonly Lazy<FaceAntiSpoof> antiSpoof = new(() => new(Path.Combine(modelDirectory, FaceAntiSpoof.ModelFileName)));
    private readonly Lazy<FaceMesh> mesh = new(() => new(Path.Combine(modelDirectory, FaceMesh.ModelFileName)));

    /// <summary>解码参考照，提取参考特征并登记新的核验会话。</summary>
    /// <param name="file">包含单张人脸的参考照编码数据。</param>
    /// <param name="requestedBackend">请求的推理后端，取 CPU 或 Vulkan。</param>
    /// <param name="cancellation">当前 HTTP 请求的取消令牌。</param>
    /// <returns>会话编号和实际使用的后端，失败时返回 HTTP 错误。</returns>
    public IResult Start(ReadOnlySpan<byte> file, string requestedBackend, CancellationToken cancellation)
    {
        long now = Stopwatch.GetTimestamp();
        foreach (var pair in sessions)
            if (Stopwatch.GetElapsedTime(pair.Value.Started, now) >= SessionLifetime)
                sessions.Remove(pair.Key);
        if (sessions.Count >= SessionLimit)
            return Results.Json(new { Message = "核验会话已满，请稍后重试。" }, statusCode: 429);

        // 创建会话前加载并校验权重，避免把模型错误延迟到首帧。
        var antiSpoofModel = antiSpoof.Value;
        var faceMesh = mesh.Value;
        VulkanModels? gpu = null;
        if (requestedBackend == "vulkan")
        {
            try { gpu = vulkan.Value; }
            catch (Exception error) when (error is NotSupportedException or InvalidOperationException)
            {
                // 响应会返回实际后端，页面据此提示回退到 CPU。
            }
        }
        var detector = gpu?.Detector ?? cpuDetector;
        var recognizer = gpu?.Recognizer ?? cpuRecognizer;
        var photo = ImageDecoder.Decode(file, "参考照片", cancellationToken: cancellation);
        var faces = detector.Detect(photo.Rgb, photo.Width, photo.Height);
        if (faces.Count != 1)
            return Results.UnprocessableEntity(new { Message = $"参考照片需要恰好一张脸，实际检出 {faces.Count} 张。" });
        if (MathF.Min(faces[0].Width, faces[0].Height) < 45)
            return Results.UnprocessableEntity(new { Message = "参考照片中的脸太小，请换一张清晰的正面照。" });

        float[] reference = recognizer.EmbedForVerification(photo.Rgb, photo.Width, photo.Height, faces[0].Landmarks, cancellation);
        string id = Guid.NewGuid().ToString("N");
        var verification = new FaceVerificationSession(detector, recognizer, antiSpoofModel, faceMesh, reference);
        sessions.Add(id, new Session(verification, Stopwatch.GetTimestamp()));
        return Results.Ok(new { Session = id, Backend = gpu is null ? "cpu" : "vulkan",
            DeviceName = gpu?.Device.DeviceName, MatchThreshold = FaceVerificationSession.MatchThreshold, Seconds = SessionLifetime.TotalSeconds });
    }

    /// <summary>解码单帧画面，返回当前动作指令或最终核验结果。</summary>
    /// <param name="id">由 <see cref="Start"/> 返回的会话编号。</param>
    /// <param name="file">摄像头帧的编码数据。</param>
    /// <param name="cancellation">当前 HTTP 请求的取消令牌。</param>
    /// <returns>带中文提示的核验状态，失败时返回 HTTP 错误。</returns>
    public IResult Frame(string id, ReadOnlySpan<byte> file, CancellationToken cancellation)
    {
        long now = Stopwatch.GetTimestamp();
        if (!sessions.TryGetValue(id, out var session))
            return Results.NotFound(new { Message = "核验会话已结束，请重新开始。" });
        if (Stopwatch.GetElapsedTime(session.Started, now) >= SessionLifetime)
        {
            sessions.Remove(id);
            return Results.Ok(new { State = "timeout", Identity = session.Verification.Similarity is null ? "pending" : "uncertain",
                session.Verification.Similarity, Liveness = "uncertain", Message = "核验超时，请重新开始。" });
        }

        var (width, height) = ImageDecoder.GetDimensions(file, "摄像头画面", MaxFramePixels);
        if (Math.Max(width, height) > 640)
            return Results.BadRequest(new { Message = "摄像头帧的长边上限为 640 像素。" });
        int length = width * height * 3;
        var buffer = ArrayPool<byte>.Shared.Rent(length);
        try
        {
            var rgb = buffer.AsSpan(0, length);
            ImageDecoder.DecodeInto(file, rgb, "摄像头画面", MaxFramePixels, cancellation);
            var result = session.Verification.ProcessFrame(rgb, width, height, cancellation);
            if (result.IsFinished)
                sessions.Remove(id);
            return Results.Ok(ToResponse(result, width, height, now));
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer, clearArray: true);
        }
    }

    /// <summary>移除指定的摄像头核验会话。</summary>
    /// <param name="id">由 <see cref="Start"/> 返回的会话编号。</param>
    /// <returns>不带响应体的 HTTP 成功响应。</returns>
    public IResult Cancel(string id)
    {
        sessions.Remove(id);
        return Results.NoContent();
    }

    /// <summary>将核心库结果转换为摄像头页面使用的 JSON 字段和中文提示。</summary>
    /// <param name="result">核心会话处理本帧后返回的状态。</param>
    /// <param name="width">解码后的画面宽度，单位为像素。</param>
    /// <param name="height">解码后的画面高度，单位为像素。</param>
    /// <param name="started">开始处理请求时由 <see cref="Stopwatch.GetTimestamp"/> 取得的时间戳。</param>
    /// <returns>供摄像头页面读取的响应对象。</returns>
    private static object ToResponse(FaceVerificationResult result, int width, int height, long started) => new
    {
        State = result.IsFinished ? "finished" : "running",
        Step = (int)result.Stage,
        Direction = result.TurnDirection,
        Action = result.ExpressionPrompt switch
        {
            FaceExpressionPrompt.Neutral => "neutral",
            FaceExpressionPrompt.CloseEyes => "closeEyes",
            FaceExpressionPrompt.OpenEyes => "openEyes",
            FaceExpressionPrompt.OpenMouth => "openMouth",
            FaceExpressionPrompt.CloseMouth => "closeMouth",
            _ => null
        },
        Identity = result.Identity switch
        {
            FaceIdentityDecision.Match => "match",
            FaceIdentityDecision.Different => "different",
            FaceIdentityDecision.Uncertain => "uncertain",
            _ => "pending"
        },
        MatchThreshold = result.Liveness == FaceLivenessDecision.Passed ? FaceVerificationSession.MatchThreshold : (float?)null,
        result.Similarity,
        Liveness = result.Liveness switch
        {
            FaceLivenessDecision.Passed => "passed",
            FaceLivenessDecision.SuspectedSpoof => "suspectedSpoof",
            _ => "pending"
        },
        result.LiveScore,
        Face = result.Face is { } face ? new CameraBox(face.X, face.Y, face.Width, face.Height) : null,
        Width = width,
        Height = height,
        result.Mesh,
        Message = result.Liveness == FaceLivenessDecision.SuspectedSpoof
            ? "防翻拍模型连续检测到疑似照片或屏幕。"
            : result.Hint switch
            {
                FaceVerificationHint.MultipleFaces => "画面中只能有一张脸，请重新正对镜头。",
                FaceVerificationHint.CenterFace => "请把脸移到画面中央。",
                FaceVerificationHint.MoveCloser => "请靠近镜头，并让整张脸留在画面里。",
                FaceVerificationHint.ShowFeatures => "请保持正脸，让眼睛和嘴巴清晰可见。",
                FaceVerificationHint.ImproveLighting => "请保持正面，尽量让脸部光照均匀。",
                FaceVerificationHint.ActionFaceChanged => "动作前后的人脸不一致，请重新正对镜头。",
                FaceVerificationHint.FinalFaceChanged => "核验开始和结束的人脸不一致，请重新正对镜头。",
                _ => null
            },
        ElapsedMs = Math.Round(Stopwatch.GetElapsedTime(started).TotalMilliseconds, 1)
    };

    private sealed record CameraBox(float X, float Y, float Width, float Height);
    private sealed record Session(FaceVerificationSession Verification, long Started);
}
