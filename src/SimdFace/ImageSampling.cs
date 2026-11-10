namespace SimdFace;

internal static class ImageSampling
{
    public static void Validate(ReadOnlySpan<byte> rgb, int width, int height)
    {
        if (width <= 0 || height <= 0 || (long)width * height != rgb.Length / 3 || rgb.Length % 3 != 0)
            throw new ArgumentException("Expected positive dimensions and tightly packed RGB24 pixels.", nameof(rgb));
    }

    public static void WritePixel(ReadOnlySpan<byte> rgb, int width, int height, float x, float y,
        Tensor target, int index, bool bgr)
    {
        if (!(x > -1 && x < width && y > -1 && y < height))
            return;
        int left = (int)MathF.Floor(x), top = (int)MathF.Floor(y);
        float fx = x - left, fy = y - top;
        for (int c = 0; c < 3; c++)
        {
            int channel = bgr ? 2 - c : c;
            float upper = Sample(rgb, width, height, left, top, channel) * (1 - fx)
                + Sample(rgb, width, height, left + 1, top, channel) * fx;
            float lower = Sample(rgb, width, height, left, top + 1, channel) * (1 - fx)
                + Sample(rgb, width, height, left + 1, top + 1, channel) * fx;
            target.Data[c * target.Plane + index] = MathF.Round(upper * (1 - fy) + lower * fy);
        }
    }

    private static byte Sample(ReadOnlySpan<byte> rgb, int width, int height, int x, int y, int channel) =>
        (uint)x < (uint)width && (uint)y < (uint)height ? rgb[(y * width + x) * 3 + channel] : (byte)0;
}
