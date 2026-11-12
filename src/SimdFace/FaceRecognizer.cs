namespace SimdFace;

/// <summary>对齐检测到的人脸并提取 SFace 身份特征。</summary>
public sealed class FaceRecognizer
{
    /// <summary>指定版本的 SFace ONNX 模型文件名。</summary>
    public const string ModelFileName = "face_recognition_sface_2021dec.onnx";
    internal const string ModelSha256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79";
    /// <summary>SFace 的余弦相似度参考阈值。</summary>
    public const float ReferenceThreshold = 0.363f;
    /// <summary>SFace 特征向量的维度。</summary>
    public const int EmbeddingSize = 128;
    /// <summary>指定版本的 SFace 模型标识。</summary>
    public const string ModelId = "sface-2021dec";
    /// <summary>六次裁剪比对流程的标识。</summary>
    public const string VerificationPipelineId = "align5-6crop-v1";
    private const int AlignedSize = 112;
    private static readonly double[] VerificationCropScales = [1.0, 0.93, 1.07];

    // SFace 对齐模板的五个关键点，顺序与 YuNet 输出一致。
    private static readonly FacePoint[] Template =
    [
        new(38.2946f, 51.6963f), new(73.5318f, 51.5014f), new(56.0252f, 71.7366f),
        new(41.5493f, 92.3655f), new(70.7299f, 92.2041f)
    ];

    private readonly ModelGraph model;

#if !NETWASM
    /// <summary>加载指定版本的 SFace 模型，使用 CPU 推理。</summary>
    /// <param name="modelPath">指定版本的 ONNX 权重文件路径。</param>
    public FaceRecognizer(string modelPath) : this(modelPath, null) { }

    /// <summary>加载指定版本的 SFace 模型，使用所选后端推理。</summary>
    /// <param name="modelPath">指定版本的 ONNX 权重文件路径。</param>
    /// <param name="vulkan">GPU 推理使用的 Vulkan 设备。传入 <see langword="null"/> 时使用 CPU。</param>
    public FaceRecognizer(string modelPath, VulkanInference? vulkan)
        : this(ModelGraph.ReadFile(modelPath), modelPath, vulkan) { }

    /// <summary>从内存加载指定版本的 ONNX 模型，使用 CPU 推理。</summary>
    /// <param name="modelBytes">指定版本 ONNX 模型的完整文件内容。</param>
    public FaceRecognizer(ReadOnlySpan<byte> modelBytes) : this(modelBytes, ModelFileName, null) { }

    private FaceRecognizer(ReadOnlySpan<byte> modelBytes, string label, VulkanInference? vulkan) =>
        model = new ModelGraph(modelBytes, label,
        ModelSha256, "data", vulkan, "fc1");
#else
    internal FaceRecognizer(ReadOnlySpan<byte> modelBytes) => model = new ModelGraph(modelBytes, "data", "fc1");
#endif

    /// <summary>对齐人脸并提取经 L2 归一化的 128 维 SFace 特征。</summary>
    /// <param name="rgb">按行连续排列的 RGB24 像素，每个像素三个字节，行尾无填充。</param>
    /// <param name="width">图像宽度，单位为像素。</param>
    /// <param name="height">图像高度，单位为像素。</param>
    /// <param name="landmarks">检测到的五个人脸关键点，用于对齐。</param>
    /// <returns>从一次对齐裁剪中提取的归一化特征向量。</returns>
    public float[] Embed(ReadOnlySpan<byte> rgb, int width, int height, FaceLandmarks landmarks) =>
        EmbedCore(rgb, width, height, landmarks, enhanced: false, CancellationToken.None);

    /// <summary>融合三张对齐裁剪图及其镜像的特征。</summary>
    /// <param name="rgb">按行连续排列的 RGB24 像素，每个像素三个字节，行尾无填充。</param>
    /// <param name="width">图像宽度，单位为像素。</param>
    /// <param name="height">图像高度，单位为像素。</param>
    /// <param name="landmarks">检测到的五个人脸关键点，用于对齐。</param>
    /// <param name="cancellationToken">在多次特征推理之间检查的取消令牌。</param>
    /// <returns>六次推理融合后的归一化特征向量。</returns>
    public float[] EmbedForVerification(ReadOnlySpan<byte> rgb, int width, int height, FaceLandmarks landmarks,
        CancellationToken cancellationToken = default) =>
        EmbedCore(rgb, width, height, landmarks, enhanced: true, cancellationToken);

    private float[] EmbedCore(ReadOnlySpan<byte> rgb, int width, int height, FaceLandmarks landmarks,
        bool enhanced, CancellationToken cancellationToken)
    {
        ImageSampling.Validate(rgb, width, height);
        ReadOnlySpan<FacePoint> points =
            [landmarks.RightEye, landmarks.LeftEye, landmarks.Nose, landmarks.RightMouth, landmarks.LeftMouth];

        double sx = 0, sy = 0, tx = 0, ty = 0;
        for (int i = 0; i < points.Length; i++)
        {
            sx += points[i].X;
            sy += points[i].Y;
            tx += Template[i].X;
            ty += Template[i].Y;
        }
        sx /= points.Length;
        sy /= points.Length;
        tx /= points.Length;
        ty /= points.Length;

        // 拟合从原图关键点到模板关键点的旋转、等比缩放和平移。
        double dot = 0, cross = 0, norm = 0;
        for (int i = 0; i < points.Length; i++)
        {
            double x = points[i].X - sx, y = points[i].Y - sy;
            double u = Template[i].X - tx, v = Template[i].Y - ty;
            dot += x * u + y * v;
            cross += x * v - y * u;
            norm += x * x + y * y;
        }
        double a = dot / norm, b = cross / norm;
        double determinant = a * a + b * b;
        if (!double.IsFinite(determinant) || determinant < 1e-12)
            throw new ArgumentException("Landmarks do not define a valid face transform.", nameof(landmarks));

        var crop = new Tensor(3, AlignedSize, AlignedSize, new float[3 * AlignedSize * AlignedSize]);
        if (!enhanced)
        {
            SampleCrop(rgb, 1);
            return Normalize(model.Run(crop)[0]);
        }

        var mirror = crop with { Data = new float[crop.Data.Length] };
        var combined = new float[EmbeddingSize];
#if !NETWASM
        Tensor[]? batch = model.UsesVulkan ? new Tensor[VerificationCropScales.Length * 2] : null;
        int batchIndex = 0;
#endif
        foreach (double scale in VerificationCropScales)
        {
            cancellationToken.ThrowIfCancellationRequested();
            // 超出原图的采样位置填零，复用缓冲区时也要清除前一次裁剪残留的像素。
            Array.Clear(crop.Data);
            SampleCrop(rgb, scale);
            ProcessCrop(crop);
            cancellationToken.ThrowIfCancellationRequested();
            for (int row = 0; row < 3 * AlignedSize; row++)
                for (int x = 0; x < AlignedSize; x++)
                    mirror.Data[row * AlignedSize + x] = crop.Data[row * AlignedSize + AlignedSize - 1 - x];
            ProcessCrop(mirror);
        }
#if !NETWASM
        if (batch is not null)
        {
            cancellationToken.ThrowIfCancellationRequested();
            foreach (float[] embedding in model.RunVulkanBatch(batch))
                AddEmbedding(embedding);
            cancellationToken.ThrowIfCancellationRequested();
        }
#endif
        return Normalize(combined);

        void ProcessCrop(Tensor sample)
        {
#if !NETWASM
            if (batch is not null)
            {
                batch[batchIndex++] = sample with { Data = (float[])sample.Data.Clone() };
                return;
            }
#endif
            AddEmbedding(model.Run(sample)[0]);
        }

        void SampleCrop(ReadOnlySpan<byte> pixels, double scale)
        {
            const double center = (AlignedSize - 1) / 2.0;
            for (int y = 0; y < AlignedSize; y++)
            for (int x = 0; x < AlignedSize; x++)
            {
                double u = (x - center) / scale + center - tx;
                double v = (y - center) / scale + center - ty;
                float sourceX = (float)(sx + (a * u + b * v) / determinant);
                float sourceY = (float)(sy + (a * v - b * u) / determinant);
                ImageSampling.WritePixel(pixels, width, height, sourceX, sourceY, crop, y * AlignedSize + x, bgr: false);
            }
        }

        void AddEmbedding(float[] embedding)
        {
            for (int i = 0; i < combined.Length; i++)
                combined[i] += embedding[i];
        }
    }

    private static float[] Normalize(float[] embedding)
    {
        float length = MathF.Sqrt(Kernels.Dot(embedding, embedding));
        if (!float.IsFinite(length) || length <= 0)
            throw new InvalidDataException("The model returned an invalid embedding.");
        for (int i = 0; i < embedding.Length; i++)
            embedding[i] /= length;
        return embedding;
    }

    /// <summary>计算两个 SFace 特征向量的余弦相似度。</summary>
    /// <param name="first">第一个特征向量。</param>
    /// <param name="second">第二个特征向量。</param>
    /// <returns>余弦相似度，取值范围为 [-1, 1]。</returns>
    /// <exception cref="ArgumentException">特征维度错误，或向量范数为零或非有限值。</exception>
    public static float CosineSimilarity(ReadOnlySpan<float> first, ReadOnlySpan<float> second)
    {
        if (first.Length != EmbeddingSize || second.Length != EmbeddingSize)
            throw new ArgumentException($"SFace embeddings must contain {EmbeddingSize} values.");
        float lengthA = MathF.Sqrt(Kernels.Dot(first, first));
        float lengthB = MathF.Sqrt(Kernels.Dot(second, second));
        if (!float.IsFinite(lengthA) || !float.IsFinite(lengthB) || lengthA <= 0 || lengthB <= 0)
            throw new ArgumentException("Embeddings must be finite and nonzero.");
        return Math.Clamp(Kernels.Dot(first, second) / lengthA / lengthB, -1, 1);
    }
}
