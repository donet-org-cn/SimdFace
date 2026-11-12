#if !NETWASM
namespace SimdFace;

/// <summary>用指定版本的 MediaPipe Face Mesh 从 RGB 图像估计 468 个人脸关键点。</summary>
public sealed class FaceMesh
{
    /// <summary>指定版本的 MediaPipe Face Mesh ONNX 模型文件名。</summary>
    public const string ModelFileName = "face_mesh_468_simdface.onnx";
    private const string ModelHash = "223be574d7c64803a6c4d22bdad84a44959b2eda4de893d59d151fe909c580e8";
    private const int Size = 192;
    private readonly ModelGraph model;

    /// <summary>加载指定版本的 MediaPipe Face Mesh 模型，使用 CPU 推理。</summary>
    /// <param name="modelPath">为 SimdFace 准备的 ONNX 权重文件路径。</param>
    public FaceMesh(string modelPath) =>
        model = new ModelGraph(ModelGraph.ReadFile(modelPath), modelPath, ModelHash, "input", null, "landmarks", "score");

    /// <summary>从按双眼位置对齐的裁剪图估计人脸关键点。</summary>
    /// <param name="rgb">按行连续排列的 RGB24 像素，每个像素三个字节，行尾无填充。</param>
    /// <param name="width">图像宽度，单位为像素。</param>
    /// <param name="height">图像高度，单位为像素。</param>
    /// <param name="face">用于定位和对齐裁剪区域的人脸框和双眼关键点。</param>
    /// <returns>468 个关键点的 XYZ 坐标，按点交错排列。X 和 Y 以原图像素为单位，Z 为模型估计的相对深度。模型未定位到人脸时返回 <see langword="null"/>。</returns>
    public float[]? Estimate(ReadOnlySpan<byte> rgb, int width, int height, DetectedFace face)
    {
        ImageSampling.Validate(rgb, width, height);
        float cx = face.X + face.Width / 2, cy = face.Y + face.Height / 2;
        float scale = 1.5f * MathF.Max(face.Width, face.Height) / Size;
        float angle = MathF.Atan2(face.Landmarks.LeftEye.Y - face.Landmarks.RightEye.Y,
            face.Landmarks.LeftEye.X - face.Landmarks.RightEye.X);
        float cos = MathF.Cos(angle) * scale, sin = MathF.Sin(angle) * scale;
        var input = new Tensor(3, Size, Size, new float[3 * Size * Size]);
        for (int row = 0; row < Size; row++)
        for (int col = 0; col < Size; col++)
        {
            float dx = col - Size / 2f, dy = row - Size / 2f;
            ImageSampling.WritePixel(rgb, width, height,
                cx + dx * cos - dy * sin, cy + dx * sin + dy * cos, input, row * Size + col, bgr: false);
        }
        for (int i = 0; i < input.Data.Length; i++)
            input.Data[i] *= 1f / 255;

        float[][] outputs = model.Run(input);
        if (outputs[1][0] <= 0 || !float.IsFinite(outputs[1][0]))
            return null;
        float[] landmarks = outputs[0];
        for (int i = 0; i < landmarks.Length; i += 3)
        {
            float x = landmarks[i] - Size / 2f, y = landmarks[i + 1] - Size / 2f;
            landmarks[i] = cx + x * cos - y * sin;
            landmarks[i + 1] = cy + x * sin + y * cos;
            landmarks[i + 2] *= scale;
        }
        return landmarks;
    }
}
#endif
