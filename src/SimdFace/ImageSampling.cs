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
        int upperLeft = PixelOffset(width, height, left, top);
        int upperRight = PixelOffset(width, height, left + 1, top);
        int lowerLeft = PixelOffset(width, height, left, top + 1);
        int lowerRight = PixelOffset(width, height, left + 1, top + 1);
        for (int c = 0; c < 3; c++)
        {
            int channel = bgr ? 2 - c : c;
            float upper = Sample(rgb, upperLeft, channel) * (1 - fx)
                + Sample(rgb, upperRight, channel) * fx;
            float lower = Sample(rgb, lowerLeft, channel) * (1 - fx)
                + Sample(rgb, lowerRight, channel) * fx;
            target.Data[c * target.Plane + index] = MathF.Round(upper * (1 - fy) + lower * fy);
        }
    }

    private static int PixelOffset(int width, int height, int x, int y) =>
        (uint)x < (uint)width && (uint)y < (uint)height ? (y * width + x) * 3 : -1;

    private static byte Sample(ReadOnlySpan<byte> rgb, int offset, int channel) =>
        offset < 0 ? (byte)0 : rgb[offset + channel];
}
