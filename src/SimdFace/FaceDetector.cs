namespace SimdFace;

public readonly record struct FacePoint(float X, float Y);

/// <summary>Five landmarks in subject-relative order and original image coordinates.</summary>
public readonly record struct FaceLandmarks(
    FacePoint RightEye, FacePoint LeftEye, FacePoint Nose, FacePoint RightMouth, FacePoint LeftMouth);

public readonly record struct DetectedFace(float X, float Y, float Width, float Height, float Score, FaceLandmarks Landmarks);

public sealed class FaceDetector
{
    public const string ModelFileName = "face_detection_yunet_2023mar.onnx";
    public const float DefaultScoreThreshold = 0.8f;
    private static readonly int[] Strides = [8, 16, 32];

    private readonly ModelGraph model;

    public FaceDetector(string modelPath) : this(modelPath, null) { }

    public FaceDetector(string modelPath, VulkanInference? vulkan) => model = new ModelGraph(modelPath,
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", "input", vulkan,
        "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32",
        "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32");

    /// <summary>Detects faces in tightly packed RGB24 data, highest confidence first.</summary>
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

        var faces = new List<DetectedFace>();
        foreach (DetectedFace candidate in candidates.OrderByDescending(face => face.Score).Take(5000))
            if (!faces.Any(face => IntersectionOverUnion(face, candidate) > 0.3f))
                faces.Add(candidate);
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
