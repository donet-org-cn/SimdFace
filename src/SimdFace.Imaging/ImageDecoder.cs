using Sdcb.SimdCodecs;
using Sdcb.SimdCodecs.Abstractions;

namespace SimdFace.Imaging;

public static class ImageDecoder
{
    public const int MaxFileBytes = 64 * 1024 * 1024;
    public const int DefaultMaxPixels = 64_000_000;

    private static readonly DecodeOptions Options = new() { ApplyOrientation = true };

    public static RgbImage ReadFile(string path)
    {
        using var file = File.OpenRead(path);
        if (file.Length > MaxFileBytes)
            throw new InvalidDataException($"Image file exceeds {MaxFileBytes / (1024 * 1024)} MiB: {path}");
        var data = new byte[(int)file.Length];
        file.ReadExactly(data);
        return Decode(data, path);
    }

    public static RgbImage Decode(ReadOnlySpan<byte> data, string label, int maxPixels = DefaultMaxPixels,
        CancellationToken cancellationToken = default)
    {
        var (width, height) = GetDimensions(data, label, maxPixels);
        var rgb = new byte[width * height * 3];
        DecodePixels(data, rgb, width, label, cancellationToken);
        return new RgbImage(rgb, width, height);
    }

    public static (int Width, int Height) GetDimensions(ReadOnlySpan<byte> data, string label,
        int maxPixels = DefaultMaxPixels)
    {
        if (!ImageCodecs.TryReadInfo(data, out ImageInfo info))
            throw new InvalidDataException($"Unsupported or invalid image: {label}");

        var (width, height) = Options.GetOutputSize(info);
        long pixels = (long)width * height;
        if (width <= 0 || height <= 0 || pixels > maxPixels || pixels > Array.MaxLength / 3)
            throw new InvalidDataException($"Image must contain between 1 and {maxPixels} pixels: {label}");
        return (width, height);
    }

    /// <summary>Decodes orientation-corrected RGB24 into a caller-owned buffer.</summary>
    public static void DecodeInto(ReadOnlySpan<byte> data, Span<byte> rgb, string label,
        int maxPixels = DefaultMaxPixels, CancellationToken cancellationToken = default)
    {
        var (width, height) = GetDimensions(data, label, maxPixels);
        DecodePixels(data, rgb[..(width * height * 3)], width, label, cancellationToken);
    }

    private static void DecodePixels(ReadOnlySpan<byte> data, Span<byte> rgb, int width, string label,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!ImageCodecs.TryDecodeInto(data, rgb, width * 3, PixelFormat.Rgb24, Options, cancellationToken))
            throw new InvalidDataException($"Could not decode image: {label}");
    }
}

public readonly record struct RgbImage(byte[] Rgb, int Width, int Height);
