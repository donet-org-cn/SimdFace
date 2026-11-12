namespace SimdFace;

/// <summary>原图中的一个点，以左上角为坐标原点。</summary>
/// <param name="X">横坐标，单位为像素。</param>
/// <param name="Y">纵坐标，单位为像素。</param>
public readonly record struct FacePoint(float X, float Y);

/// <summary>原图中的五个人脸关键点，左右以被测人自身为准。</summary>
/// <param name="RightEye">被测人的右眼中心。</param>
/// <param name="LeftEye">被测人的左眼中心。</param>
/// <param name="Nose">鼻尖。</param>
/// <param name="RightMouth">被测人的右嘴角。</param>
/// <param name="LeftMouth">被测人的左嘴角。</param>
public readonly record struct FaceLandmarks(
    FacePoint RightEye, FacePoint LeftEye, FacePoint Nose, FacePoint RightMouth, FacePoint LeftMouth);

/// <summary>一次人脸检测的结果，包含原图中的人脸框和关键点，以及检测置信度。</summary>
/// <param name="X">人脸框左边缘的横坐标，单位为像素。</param>
/// <param name="Y">人脸框上边缘的纵坐标，单位为像素。</param>
/// <param name="Width">人脸框宽度，单位为像素。</param>
/// <param name="Height">人脸框高度，单位为像素。</param>
/// <param name="Score">YuNet 检测置信度。</param>
/// <param name="Landmarks">用于人脸对齐的五个关键点。</param>
public readonly record struct DetectedFace(float X, float Y, float Width, float Height, float Score, FaceLandmarks Landmarks);

/// <summary>用 YuNet 检测人脸及五个关键点。</summary>
public sealed class FaceDetector
{
    /// <summary>指定版本的 YuNet ONNX 模型文件名。</summary>
    public const string ModelFileName = "face_detection_yunet_2023mar.onnx";
    internal const string ModelSha256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4";
    /// <summary>检测置信度的默认下限。</summary>
    public const float DefaultScoreThreshold = 0.8f;
    private static readonly int[] Strides = [8, 16, 32];

    private readonly ModelGraph model;

#if !NETWASM
    /// <summary>加载指定版本的 YuNet 模型，使用 CPU 推理。</summary>
    /// <param name="modelPath">指定版本的 ONNX 权重文件路径。</param>
    public FaceDetector(string modelPath) : this(modelPath, null) { }

    /// <summary>加载指定版本的 YuNet 模型，使用所选后端推理。</summary>
    /// <param name="modelPath">指定版本的 ONNX 权重文件路径。</param>
    /// <param name="vulkan">GPU 推理使用的 Vulkan 设备。传入 <see langword="null"/> 时使用 CPU。</param>
    public FaceDetector(string modelPath, VulkanInference? vulkan)
        : this(ModelGraph.ReadFile(modelPath), modelPath, vulkan) { }

    /// <summary>从内存加载指定版本的 ONNX 模型，使用 CPU 推理。</summary>
    /// <param name="modelBytes">指定版本 ONNX 模型的完整文件内容。</param>
    public FaceDetector(ReadOnlySpan<byte> modelBytes) : this(modelBytes, ModelFileName, null) { }

    private FaceDetector(ReadOnlySpan<byte> modelBytes, string label, VulkanInference? vulkan) =>
        model = new ModelGraph(modelBytes, label,
        ModelSha256, "input", vulkan,
        "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32",
        "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32");
#else
    internal FaceDetector(ReadOnlySpan<byte> modelBytes) => model = new ModelGraph(modelBytes, "input",
        "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32",
        "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32");
#endif

    /// <summary>在 RGB24 图像中检测人脸。</summary>
    /// <param name="rgb">按行连续排列的 RGB24 像素，每个像素三个字节，行尾无填充。</param>
    /// <param name="width">图像宽度，单位为像素。</param>
    /// <param name="height">图像高度，单位为像素。</param>
    /// <param name="scoreThreshold">检测置信度的下限，取值范围为 (0, 1]。</param>
    /// <returns>按检测置信度从高到低排列的人脸结果。</returns>
    /// <exception cref="ArgumentOutOfRangeException">置信度阈值超出允许的范围。</exception>
    public IReadOnlyList<DetectedFace> Detect(ReadOnlySpan<byte> rgb, int width, int height,
        float scoreThreshold = DefaultScoreThreshold)
    {
        ImageSampling.Validate(rgb, width, height);
        if (!(scoreThreshold > 0 && scoreThreshold <= 1))
            throw new ArgumentOutOfRangeException(nameof(scoreThreshold));

        double scale = Math.Min(1, 640d / Math.Max(width, height));
        int resizedWidth = Math.Max(1, (int)Math.Round(width * scale));
        int resizedHeight = Math.Max(1, (int)Math.Round(height * scale));
        int paddedWidth = (resizedWidth + 31) / 32 * 32;
        int paddedHeight = (resizedHeight + 31) / 32 * 32;
        var input = new Tensor(3, paddedHeight, paddedWidth, new float[3 * paddedWidth * paddedHeight]);
        float scaleX = (float)width / resizedWidth, scaleY = (float)height / resizedHeight;
        for (int y = 0; y < resizedHeight; y++)
        for (int x = 0; x < resizedWidth; x++)
        {
            int index = y * paddedWidth + x;
            if (scale == 1)
            {
                int source = (y * width + x) * 3;
                input.Data[index] = rgb[source + 2];
                input.Data[input.Plane + index] = rgb[source + 1];
                input.Data[2 * input.Plane + index] = rgb[source];
            }
            else
            {
                float sourceX = Math.Clamp((x + 0.5f) * scaleX - 0.5f, 0, width - 1);
                float sourceY = Math.Clamp((y + 0.5f) * scaleY - 0.5f, 0, height - 1);
                ImageSampling.WritePixel(rgb, width, height, sourceX, sourceY, input, index, bgr: true);
            }
        }

        float[][] predictions = model.Run(input);
        var candidates = new List<DetectedFace>();
        for (int level = 0; level < Strides.Length; level++)
        {
            int stride = Strides[level];
            int columns = paddedWidth / stride;
            float[] cls = predictions[level], obj = predictions[level + 3];
            float[] box = predictions[level + 6], points = predictions[level + 9];
            for (int i = 0; i < cls.Length; i++)
            {
                float score = MathF.Sqrt(cls[i] * obj[i]);
                if (!(score >= scoreThreshold))
                    continue;
                int row = i / columns, col = i % columns;
                float cx = (col + box[i * 4]) * stride;
                float cy = (row + box[i * 4 + 1]) * stride;
                if (cx < 0 || cx >= resizedWidth || cy < 0 || cy >= resizedHeight)
                    continue;
                float w = MathF.Exp(box[i * 4 + 2]) * stride;
                float h = MathF.Exp(box[i * 4 + 3]) * stride;
                if (!float.IsFinite(w) || !float.IsFinite(h) || w <= 0 || h <= 0)
                    continue;
                FacePoint PointAt(int offset) => new((col + points[i * 10 + offset]) * stride * scaleX,
                    (row + points[i * 10 + offset + 1]) * stride * scaleY);
                var landmarks = new FaceLandmarks(PointAt(0), PointAt(2), PointAt(4), PointAt(6), PointAt(8));
                candidates.Add(new DetectedFace((cx - w / 2) * scaleX, (cy - h / 2) * scaleY,
                    w * scaleX, h * scaleY, score, landmarks));
            }
        }

        var order = new int[candidates.Count];
        for (int i = 0; i < order.Length; i++)
            order[i] = i;
        Array.Sort(order, (left, right) =>
        {
            int scoreOrder = candidates[right].Score.CompareTo(candidates[left].Score);
            return scoreOrder == 0 ? left.CompareTo(right) : scoreOrder;
        });

        var faces = new List<DetectedFace>();
        int count = Math.Min(order.Length, 5000);
        for (int i = 0; i < count; i++)
        {
            DetectedFace candidate = candidates[order[i]];
            bool overlaps = false;
            foreach (var face in faces)
            {
                overlaps = IntersectionOverUnion(face, candidate) > 0.3f;
                if (overlaps)
                    break;
            }
            if (!overlaps)
                faces.Add(candidate);
        }
        return faces;
    }

    private static float IntersectionOverUnion(DetectedFace a, DetectedFace b)
    {
        float w = MathF.Max(0, MathF.Min(a.X + a.Width, b.X + b.Width) - MathF.Max(a.X, b.X));
        float h = MathF.Max(0, MathF.Min(a.Y + a.Height, b.Y + b.Height) - MathF.Max(a.Y, b.Y));
        float overlap = w * h;
        return overlap / (a.Width * a.Height + b.Width * b.Height - overlap);
    }
}
