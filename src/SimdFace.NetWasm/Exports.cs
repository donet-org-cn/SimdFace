using System.Runtime.InteropServices.JavaScript;

namespace SimdFace;

public static class Exports
{
    private const int MaxPixels = 64_000_000;
    private static FaceDetector? detector;
    private static FaceRecognizer? recognizer;
    private static byte[]? buffer;
    private static int written;

    [JSExport]
    public static string ModelHash(bool recognition) => recognition ? FaceRecognizer.ModelSha256 : FaceDetector.ModelSha256;

    [JSExport]
    public static void BeginBuffer(int length)
    {
        if (length <= 0 || length > MaxPixels * 3)
            throw new ArgumentOutOfRangeException(nameof(length));
        buffer = new byte[length];
        written = 0;
    }

    [JSExport]
    public static void AppendBytes(byte[] bytes)
    {
        if (buffer is null || bytes.Length > buffer.Length - written)
            throw new InvalidOperationException("Invalid pixel or model data length.");
        Array.Copy(bytes, 0, buffer, written, bytes.Length);
        written += bytes.Length;
    }

    [JSExport]
    public static void LoadModel(bool recognition)
    {
        // The browser host checks the model's pinned SHA-256 before transferring it.
        var model = TakeBuffer();
        if (recognition)
            recognizer = new FaceRecognizer(model);
        else
            detector = new FaceDetector(model);
    }

    [JSExport]
    public static float[] AnalyzeImage(int width, int height, float score, bool extract)
    {
        var pixels = TakeBuffer();
        if (detector is null || (extract && recognizer is null))
            throw new InvalidOperationException("请先加载模型。");
        var faces = detector.Detect(pixels, width, height, score);
        float[]? feature = extract && faces.Count == 1
            ? recognizer!.EmbedForVerification(pixels, width, height, faces[0].Landmarks)
            : null;
        var output = new float[1 + faces.Count * 15 + (feature?.Length ?? 0)];
        output[0] = faces.Count;
        int i = 1;
        foreach (var face in faces)
        {
            output[i++] = face.X;
            output[i++] = face.Y;
            output[i++] = face.Width;
            output[i++] = face.Height;
            output[i++] = face.Score;
            var points = face.Landmarks;
            WritePoint(points.RightEye);
            WritePoint(points.LeftEye);
            WritePoint(points.Nose);
            WritePoint(points.RightMouth);
            WritePoint(points.LeftMouth);
        }
        if (feature is not null)
            Array.Copy(feature, 0, output, i, feature.Length);
        return output;

        void WritePoint(FacePoint point)
        {
            output[i++] = point.X;
            output[i++] = point.Y;
        }
    }

    [JSExport]
    public static float CompareVectors(float[] first, float[] second) => FaceRecognizer.CosineSimilarity(first, second);

    private static byte[] TakeBuffer()
    {
        var result = buffer;
        buffer = null;
        if (result is null || result.Length != written)
            throw new InvalidOperationException("Incomplete pixel or model data.");
        return result;
    }
}
