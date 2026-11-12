using System.Diagnostics;
using System.Security.Cryptography;

namespace SimdFace;

/// <summary>跟踪画面中的单张人脸，执行活体检测，并与参考照片的 SFace 特征比对。</summary>
/// <remarks>
/// 同一个会话应按采集顺序串行调用 <see cref="ProcessFrame"/>。
/// 调用方负责采集、解码和会话超时。已加载的模型可供新会话复用。
/// 同一次核验应保持画面方向和镜像状态一致。动作持续时间按每次调用时的单调时钟计算。
/// </remarks>
public sealed class FaceVerificationSession
{
    /// <summary>两次正脸特征融合后，与参考特征判定为匹配所需的最低相似度。</summary>
    public const float MatchThreshold = 0.61f;

    private static readonly TimeSpan MaxTrackingGap = TimeSpan.FromMilliseconds(1400);
    private static readonly TimeSpan BlinkTrackingGrace = TimeSpan.FromMilliseconds(250);
    private static readonly TimeSpan BlinkReopenGrace = TimeSpan.FromMilliseconds(700);
    private static readonly TimeSpan HoldTime = TimeSpan.FromMilliseconds(120);
    private const float MinLiveScore = 0.55f;
    private const float StrongSpoofScore = 0.25f;
    private const float TurnAmount = 0.20f;
    private const float ContinuityThreshold = 0.68f;
    private const float MinimumFrameEvidence = 0.54f;

    private readonly FaceDetector detector;
    private readonly FaceRecognizer recognizer;
    private readonly FaceAntiSpoof antiSpoof;
    private readonly FaceMesh mesh;
    private readonly float[] reference;
    private int firstDirection = NewDirection();
    private FacialAction action = FacialAction.CreateRandom();
    private FaceVerificationStage stage;
    private long? heldSince;
    private long? lastTrackedAt;
    private long? lastScoreAt;
    private DetectedFace? lastFace;
    private float startYaw;
    private float[]? startEmbedding;
    private float? liveScore;
    private int lowLiveFrames;
    private int goodLiveFrames;
    private bool finished;

    /// <summary>创建一次核验会话，并为本次核验随机生成动作指令。</summary>
    /// <param name="detector">用于检测和跟踪单张人脸的 YuNet 检测器。</param>
    /// <param name="recognizer">用于身份比对的 SFace 特征提取器。</param>
    /// <param name="antiSpoof">已加载的 MiniFASNetV2 模型，用于对接近正脸的画面做防翻拍评分。</param>
    /// <param name="mesh">已加载的 468 点人脸网格模型，用于判定眼睛和嘴部动作。</param>
    /// <param name="referenceEmbedding">参考照片的人脸特征，由
    /// <see cref="FaceRecognizer.EmbedForVerification"/> 提取。会话会复制该向量。</param>
    /// <exception cref="ArgumentException">参考特征的维度错误，或向量范数为零或非有限值。</exception>
    public FaceVerificationSession(FaceDetector detector, FaceRecognizer recognizer, FaceAntiSpoof antiSpoof,
        FaceMesh mesh, ReadOnlySpan<float> referenceEmbedding)
    {
        ArgumentNullException.ThrowIfNull(detector);
        ArgumentNullException.ThrowIfNull(recognizer);
        ArgumentNullException.ThrowIfNull(antiSpoof);
        ArgumentNullException.ThrowIfNull(mesh);
        if (referenceEmbedding.Length != FaceRecognizer.EmbeddingSize)
            throw new ArgumentException($"Expected {FaceRecognizer.EmbeddingSize} SFace values.", nameof(referenceEmbedding));
        float norm = Kernels.Dot(referenceEmbedding, referenceEmbedding);
        if (!float.IsFinite(norm) || norm <= 0)
            throw new ArgumentException("Expected a finite, nonzero SFace verification embedding.", nameof(referenceEmbedding));

        this.detector = detector;
        this.recognizer = recognizer;
        this.antiSpoof = antiSpoof;
        this.mesh = mesh;
        reference = referenceEmbedding.ToArray();
    }

    /// <summary>最近一次与参考特征的相似度。尚未比对时为 <see langword="null"/>。</summary>
    public float? Similarity { get; private set; }

    /// <summary>处理一帧采集画面，返回下一步动作提示或核验结论。</summary>
    /// <param name="rgb">按行连续排列的 RGB24 像素，每个像素三个字节，行尾无填充。</param>
    /// <param name="width">画面宽度，单位为像素。</param>
    /// <param name="height">画面高度，单位为像素。</param>
    /// <param name="cancellationToken">在开始推理前或多次特征推理之间检查的取消令牌。</param>
    /// <returns>当前动作阶段，以及本帧可用的人脸位置、相似度和活体结论。</returns>
    /// <exception cref="ArgumentException">画面尺寸无效，或 RGB 数据长度与尺寸不符。</exception>
    /// <exception cref="InvalidOperationException">会话已得出最终结论。</exception>
    /// <exception cref="OperationCanceledException">调用方已请求取消处理。</exception>
    public FaceVerificationResult ProcessFrame(ReadOnlySpan<byte> rgb, int width, int height,
        CancellationToken cancellationToken = default)
    {
        if (finished)
            throw new InvalidOperationException("Verification is complete. Create a new session for the next attempt.");
        cancellationToken.ThrowIfCancellationRequested();
        long now = Stopwatch.GetTimestamp();
        var faces = detector.Detect(rgb, width, height);
        DetectedFace? face = faces.Count == 1 ? faces[0] : null;
        if (lastTrackedAt is long last && Stopwatch.GetElapsedTime(last, now) > MaxTrackingGap)
            Reset();
        if (face is not DetectedFace current)
        {
            if (faces.Count == 0 && stage == FaceVerificationStage.Expression && action.IsBlinkInProgress &&
                lastFace is DetectedFace previousFace && lastTrackedAt is long trackedAt &&
                Stopwatch.GetElapsedTime(trackedAt, now) <=
                    (action.WaitingForEyeClosure ? BlinkTrackingGrace : BlinkReopenGrace))
            {
                // 闭眼时 YuNet 可能短暂丢失检测。闭眼阶段可用最近一次检测到的区域估计网格，
                // 随后必须重新检测到人脸才能完成睁眼阶段。
                if (action.WaitingForEyeClosure &&
                    mesh.Estimate(rgb, width, height, previousFace) is { } points)
                    action.Observe(points, now);
                return Result(face);
            }
            if (faces.Count > 1)
                Reset();
            InterruptPose();
            return Result(face, faces.Count > 1 ? FaceVerificationHint.MultipleFaces : FaceVerificationHint.CenterFace);
        }

        if (MathF.Min(current.Width, current.Height) < Math.Max(75, Math.Min(width, height) * 0.24f) ||
            current.X < 2 || current.Y < 2 || current.X + current.Width > width - 2 || current.Y + current.Height > height - 2)
        {
            InterruptPose();
            return Result(face, FaceVerificationHint.MoveCloser);
        }

        if (lastFace is DetectedFace previous && !SameTrack(previous, current))
            Reset();
        lastFace = current;
        lastTrackedAt = now;

        float yaw = EstimateYaw(current.Landmarks);
        // 在接近正脸时定期运行防翻拍模型，覆盖小角度倾斜的画面。
        bool scoreFrame = (stage is FaceVerificationStage.Frontal or FaceVerificationStage.FinalFrontal) && MathF.Abs(yaw) < 0.40f &&
            (lastScoreAt is not long scored || Stopwatch.GetElapsedTime(scored, now) >= HoldTime);
        if (scoreFrame)
        {
            liveScore = antiSpoof.LiveScore(rgb, width, height, current);
            lastScoreAt = now;
            lowLiveFrames = liveScore <= StrongSpoofScore ? lowLiveFrames + 1 : 0;
            if (lowLiveFrames >= 2)
            {
                Similarity ??= Compare(rgb, width, height, current, cancellationToken);
                return Result(face, liveness: FaceLivenessDecision.SuspectedSpoof, identity: SingleFrameIdentity(Similarity));
            }
        }

        float relativeYaw = yaw - startYaw;
        bool aligned = stage switch
        {
            FaceVerificationStage.Frontal => MathF.Abs(yaw) < 0.18f,
            FaceVerificationStage.Turn => relativeYaw * firstDirection >= TurnAmount,
            FaceVerificationStage.Expression when action.IsBlinkInProgress => MathF.Abs(relativeYaw) < 0.25f && MathF.Abs(yaw) < 0.27f,
            _ => MathF.Abs(relativeYaw) < 0.16f && MathF.Abs(yaw) < 0.20f
        };
        if (!aligned || !float.IsFinite(yaw))
        {
            InterruptPose();
            goodLiveFrames = 0;
            return Result(face);
        }

        if (stage == FaceVerificationStage.Expression)
        {
            var points = mesh.Estimate(rgb, width, height, current);
            if (points is null)
            {
                action.Reset();
                return Result(face, FaceVerificationHint.ShowFeatures);
            }
            if (!action.Observe(points, now))
                return Result(face);
        }
        else
        {
            if (heldSince is not long held)
            {
                heldSince = now;
                return Result(face);
            }
            if (Stopwatch.GetElapsedTime(held, now) < HoldTime)
                return Result(face);
        }

        if (stage is FaceVerificationStage.Turn or FaceVerificationStage.Expression)
        {
            // 动作完成后提取一次特征，确认做动作的人与起始正脸一致。
            var actionFace = recognizer.Embed(rgb, width, height, current.Landmarks);
            if (FaceRecognizer.CosineSimilarity(startEmbedding!, actionFace) < FaceRecognizer.ReferenceThreshold)
            {
                Reset();
                return Result(face, FaceVerificationHint.ActionFaceChanged);
            }
            Advance();
            return Result(face);
        }

        if (!scoreFrame)
            return Result(face);
        if (liveScore is not float live || !float.IsFinite(live) || live < MinLiveScore)
        {
            goodLiveFrames = 0;
            return Result(face, FaceVerificationHint.ImproveLighting);
        }
        if (++goodLiveFrames < 2)
            return Result(face);

        var embedding = recognizer.EmbedForVerification(rgb, width, height, current.Landmarks, cancellationToken);
        Similarity = FaceRecognizer.CosineSimilarity(reference, embedding);
        if (stage == FaceVerificationStage.Frontal)
        {
            startYaw = yaw;
            startEmbedding = embedding;
            Advance();
            return Result(face,
                landmarks: mesh.Estimate(rgb, width, height, current));
        }

        if (startEmbedding is not { } start || FaceRecognizer.CosineSimilarity(start, embedding) < ContinuityThreshold)
        {
            Reset();
            return Result(face, FaceVerificationHint.FinalFaceChanged);
        }

        // 确认两次正脸是同一张脸后，融合两次采样的特征再与参考照比对。
        float weakerScore = MathF.Min(FaceRecognizer.CosineSimilarity(reference, start), Similarity.Value);
        Span<float> template = stackalloc float[FaceRecognizer.EmbeddingSize];
        for (int i = 0; i < template.Length; i++)
            template[i] = start[i] + embedding[i];
        float score = FaceRecognizer.CosineSimilarity(reference, template);
        Similarity = score;
        FaceIdentityDecision identity = score >= MatchThreshold && weakerScore >= MinimumFrameEvidence
            ? FaceIdentityDecision.Match : score < FaceRecognizer.ReferenceThreshold
                ? FaceIdentityDecision.Different : FaceIdentityDecision.Uncertain;

        var landmarks = mesh.Estimate(rgb, width, height, current);
        return Result(face, landmarks: landmarks, liveness: FaceLivenessDecision.Passed, identity: identity);
    }

    private float Compare(ReadOnlySpan<byte> rgb, int width, int height, DetectedFace face,
        CancellationToken cancellationToken) => FaceRecognizer.CosineSimilarity(reference,
        recognizer.EmbedForVerification(rgb, width, height, face.Landmarks, cancellationToken));

    private FaceVerificationResult Result(DetectedFace? face, FaceVerificationHint hint = FaceVerificationHint.None,
        float[]? landmarks = null, FaceLivenessDecision liveness = FaceLivenessDecision.Pending,
        FaceIdentityDecision identity = FaceIdentityDecision.Pending)
    {
        finished = liveness != FaceLivenessDecision.Pending;
        return new FaceVerificationResult(stage, stage == FaceVerificationStage.Turn ? firstDirection : 0,
            stage == FaceVerificationStage.Expression ? action.Prompt : null,
            identity, liveness, Similarity, liveScore, face, landmarks, hint);
    }

    private static FaceIdentityDecision SingleFrameIdentity(float? score) => score switch
    {
        null => FaceIdentityDecision.Pending,
        float value when value >= ContinuityThreshold => FaceIdentityDecision.Match,
        float value when value < FaceRecognizer.ReferenceThreshold => FaceIdentityDecision.Different,
        _ => FaceIdentityDecision.Uncertain
    };

    // 把鼻尖投影到双眼连线，减少画面内歪头对左右方向判断的影响。
    private static float EstimateYaw(FaceLandmarks points)
    {
        var right = points.RightEye;
        var left = points.LeftEye;
        float dx = left.X - right.X, dy = left.Y - right.Y;
        return ((points.Nose.X - (left.X + right.X) / 2) * dx +
                (points.Nose.Y - (left.Y + right.Y) / 2) * dy) / (dx * dx + dy * dy);
    }

    private void InterruptPose()
    {
        heldSince = null;
        if (stage == FaceVerificationStage.Expression)
            action.Reset();
    }

    private static bool SameTrack(DetectedFace a, DetectedFace b)
    {
        float overlapWidth = MathF.Min(a.X + a.Width, b.X + b.Width) - MathF.Max(a.X, b.X);
        float overlapHeight = MathF.Min(a.Y + a.Height, b.Y + b.Height) - MathF.Max(a.Y, b.Y);
        return overlapWidth > 0 && overlapHeight > 0 &&
            overlapWidth * overlapHeight >= MathF.Min(a.Width * a.Height, b.Width * b.Height) * 0.45f;
    }

    private void Advance()
    {
        stage++;
        heldSince = lastScoreAt = null;
        lowLiveFrames = 0;
        goodLiveFrames = 0;
    }

    // 跟踪中断或人脸不一致时，清空已有证据并重新生成动作指令。
    private void Reset()
    {
        firstDirection = NewDirection();
        action = FacialAction.CreateRandom();
        stage = FaceVerificationStage.Frontal;
        heldSince = lastTrackedAt = lastScoreAt = null;
        lastFace = null;
        startYaw = 0;
        startEmbedding = null;
        Similarity = liveScore = null;
        lowLiveFrames = goodLiveFrames = 0;
    }

    private static int NewDirection() => RandomNumberGenerator.GetInt32(2) == 0 ? -1 : 1;
}
