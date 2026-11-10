namespace SimdFace;

public sealed class FaceRecognizer
{
    public const string ModelFileName = "face_recognition_sface_2021dec.onnx";
    public const float ReferenceThreshold = 0.363f;
    public const int EmbeddingSize = 128;
    private const int AlignedSize = 112;
    private static readonly double[] VerificationCropScales = [1.0, 0.93, 1.07];

    // SFace's five-point reference geometry, in the same order as YuNet.
    private static readonly FacePoint[] Template =
    [
        new(38.2946f, 51.6963f), new(73.5318f, 51.5014f), new(56.0252f, 71.7366f),
        new(41.5493f, 92.3655f), new(70.7299f, 92.2041f)
    ];

    private readonly ModelGraph model;

    public FaceRecognizer(string modelPath) : this(modelPath, null) { }

    public FaceRecognizer(string modelPath, VulkanInference? vulkan) => model = new ModelGraph(modelPath,
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79", "data", vulkan, "fc1");

    /// <summary>Aligns the face and returns an L2-normalized, 128-dimensional SFace embedding.</summary>
    public float[] Embed(ReadOnlySpan<byte> rgb, int width, int height, FaceLandmarks landmarks) =>
        EmbedCore(rgb, width, height, landmarks, enhanced: false, CancellationToken.None);

    /// <summary>Combines three aligned crops and their mirrors into one normalized SFace embedding.</summary>
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

        // Fit a rotation, uniform scale and translation from source to template.
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
        foreach (double scale in VerificationCropScales)
        {
            cancellationToken.ThrowIfCancellationRequested();
            // Out-of-frame pixels sample as zero, including when a wider crop follows an earlier crop.
            Array.Clear(crop.Data);
            SampleCrop(rgb, scale);
            AddEmbedding(model.Run(crop)[0]);
            cancellationToken.ThrowIfCancellationRequested();
            for (int row = 0; row < 3 * AlignedSize; row++)
                for (int x = 0; x < AlignedSize; x++)
                    mirror.Data[row * AlignedSize + x] = crop.Data[row * AlignedSize + AlignedSize - 1 - x];
            AddEmbedding(model.Run(mirror)[0]);
        }
        return Normalize(combined);

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
