namespace SimdFace;

/// <summary>一次活体核验的当前阶段。</summary>
public enum FaceVerificationStage
{
    /// <summary>采集起始正脸特征，并进行防翻拍检测。</summary>
    Frontal,
    /// <summary>检查是否向随机指定的一侧转头。</summary>
    Turn,
    /// <summary>回到正脸后，检查闭眼睁眼或张嘴合嘴动作。</summary>
    Expression,
    /// <summary>采集最后一次正脸特征，并与参考照比对。</summary>
    FinalFrontal
}

/// <summary>眼睛或嘴部动作阶段的提示。</summary>
public enum FaceExpressionPrompt
{
    /// <summary>正对摄像头并保持自然表情，供会话采集基线。</summary>
    Neutral,
    /// <summary>短暂闭上双眼。</summary>
    CloseEyes,
    /// <summary>重新睁开双眼。</summary>
    OpenEyes,
    /// <summary>张嘴并短暂保持。</summary>
    OpenMouth,
    /// <summary>重新合嘴。</summary>
    CloseMouth
}

/// <summary>当前人脸与参考照的身份比对结论。</summary>
public enum FaceIdentityDecision
{
    /// <summary>正在采集身份比对所需的样本。</summary>
    Pending,
    /// <summary>采集到的人脸特征与参考特征匹配。</summary>
    Match,
    /// <summary>相似度低于身份比对的参考阈值，判定为不同人。</summary>
    Different,
    /// <summary>现有证据不足以确认是同一人。</summary>
    Uncertain
}

/// <summary>当前会话的活体检测结论。</summary>
public enum FaceLivenessDecision
{
    /// <summary>活体检测仍在进行。</summary>
    Pending,
    /// <summary>通过防翻拍检测和指定动作检查。</summary>
    Passed,
    /// <summary>防翻拍模型在多个采样帧中检测到疑似翻拍。</summary>
    SuspectedSpoof
}

/// <summary>调用方可用于向用户展示补充指引的状态。</summary>
public enum FaceVerificationHint
{
    /// <summary>继续按当前动作指令操作。</summary>
    None,
    /// <summary>画面检测到多张脸，应仅保留被测人的脸。</summary>
    MultipleFaces,
    /// <summary>将人脸移到画面中央。</summary>
    CenterFace,
    /// <summary>靠近镜头，并让整张脸留在画面中。</summary>
    MoveCloser,
    /// <summary>正对摄像头，并让眼睛和嘴部清晰可见。</summary>
    ShowFeatures,
    /// <summary>正对摄像头，尽量让脸部光照均匀。</summary>
    ImproveLighting,
    /// <summary>动作前后的人脸不一致，从正脸阶段重新核验。</summary>
    ActionFaceChanged,
    /// <summary>起始和结束的人脸不一致，从正脸阶段重新核验。</summary>
    FinalFaceChanged
}

/// <summary>核验会话处理一帧画面后的结果。</summary>
/// <param name="Stage">处理本帧后的当前核验阶段。</param>
/// <param name="TurnDirection">正数表示向被测人自己的左侧转头，负数表示向右侧转头，
/// 其他阶段为零。</param>
/// <param name="ExpressionPrompt">眼睛或嘴部动作阶段的指令。其他阶段为 <see langword="null"/>。</param>
/// <param name="Identity">与参考照的身份比对结论。</param>
/// <param name="Liveness">独立的活体检测结论。</param>
/// <param name="Similarity">最近一次与参考特征的余弦相似度，尚未比对时为 <see langword="null"/>。</param>
/// <param name="LiveScore">最近一次防翻拍模型的活体类别分数，尚未评分时为 <see langword="null"/>。</param>
/// <param name="Face">本帧检测到的单张人脸，检测不到或检测到多张时为 <see langword="null"/>。</param>
/// <param name="Mesh">正脸采样时可用的 468 个交错排列的 XYZ 坐标。</param>
/// <param name="Hint">当前画面的补充提示。</param>
public readonly record struct FaceVerificationResult(
    FaceVerificationStage Stage,
    int TurnDirection,
    FaceExpressionPrompt? ExpressionPrompt,
    FaceIdentityDecision Identity,
    FaceLivenessDecision Liveness,
    float? Similarity,
    float? LiveScore,
    DetectedFace? Face,
    float[]? Mesh,
    FaceVerificationHint Hint)
{
    /// <summary>会话是否已完成。完成后应创建新会话再提交帧。</summary>
    public bool IsFinished => Liveness != FaceLivenessDecision.Pending;
}
