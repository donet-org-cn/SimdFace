#if !NETWASM
namespace SimdFace;

/// <summary>用指定版本的 MiniFASNetV2 对人脸进行防翻拍评分。</summary>
public sealed class FaceAntiSpoof
{
    /// <summary>指定版本的 MiniFASNetV2 ONNX 模型文件名。</summary>
    public const string ModelFileName = "minifasnet_v2_simdface.onnx";
    private const string ModelHash = "a9aab305543784f71531e82cebd1a83d309f1cfcb2c8559fdf3832503facdd3a";
    private const int Size = 80;
    private readonly ModelGraph model;

    /// <summary>加载指定版本的 MiniFASNetV2 模型，使用 CPU 推理。</summary>
    /// <param name="modelPath">为 SimdFace 准备的 ONNX 权重文件路径。</param>
    public FaceAntiSpoof(string modelPath) =>
        model = new ModelGraph(ModelGraph.ReadFile(modelPath), modelPath, ModelHash, "input", null, "output");

    /// <summary>根据人脸及周边区域计算活体分数。</summary>
    /// <param name="rgb">按行连续排列的 RGB24 像素，每个像素三个字节，行尾无填充。</param>
    /// <param name="width">图像宽度，单位为像素。</param>
    /// <param name="height">图像高度，单位为像素。</param>
    /// <param name="face">在输入图像中检测到的人脸框。</param>
    /// <returns>活体类别的 softmax 分数，取值范围为 [0, 1]。</returns>
    public float LiveScore(ReadOnlySpan<byte> rgb, int width, int height, DetectedFace face)
    {
        ImageSampling.Validate(rgb, width, height);
        float scale = MathF.Min(2.7f, MathF.Min((width - 1) / face.Width, (height - 1) / face.Height));
        float boxWidth = face.Width * scale, boxHeight = face.Height * scale;
        float x = Math.Clamp(face.X + (face.Width - boxWidth) / 2, 0, MathF.Max(0, width - 1 - boxWidth));
        float y = Math.Clamp(face.Y + (face.Height - boxHeight) / 2, 0, MathF.Max(0, height - 1 - boxHeight));
        int left = (int)x, top = (int)y;
        int right = (int)(x + boxWidth), bottom = (int)(y + boxHeight);
        float sx = (right - left + 1f) / Size, sy = (bottom - top + 1f) / Size;
        var input = new Tensor(3, Size, Size, new float[3 * Size * Size]);
        for (int row = 0; row < Size; row++)
        for (int col = 0; col < Size; col++)
            ImageSampling.WritePixel(rgb, width, height,
                Math.Clamp(left + (col + 0.5f) * sx - 0.5f, left, right),
                Math.Clamp(top + (row + 0.5f) * sy - 0.5f, top, bottom),
                input, row * Size + col, bgr: true);

        float[] logits = model.Run(input)[0];
        float max = MathF.Max(logits[1], MathF.Max(logits[0], logits[2]));
        float live = MathF.Exp(logits[1] - max);
        return live / (MathF.Exp(logits[0] - max) + live + MathF.Exp(logits[2] - max));
    }
}
#endif
