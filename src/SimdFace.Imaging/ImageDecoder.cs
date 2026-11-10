using Sdcb.SimdCodecs;
using Sdcb.SimdCodecs.Abstractions;

namespace SimdFace.Imaging;

public static class ImageDecoder
{
    public static RgbImage ReadFile(string path)
    {
        using var file = File.OpenRead(path);
        if (file.Length > 64 * 1024 * 1024)
            throw new InvalidDataException($"Image file exceeds 64 MiB: {path}");
        var data = new byte[(int)file.Length];
        file.ReadExactly(data);
        return Decode(data, path);
    }

    public static RgbImage Decode(ReadOnlySpan<byte> data, string label, int maxPixels = 40_000_000,
        CancellationToken cancellationToken = default)
    {
        if (!ImageCodecs.TryReadInfo(data, out ImageInfo info))
            throw new InvalidDataException($"Unsupported or invalid image: {label}");

        var options = new DecodeOptions { ApplyOrientation = true };
        var (width, height) = options.GetOutputSize(info);
        if (width <= 0 || height <= 0 || (long)width * height > maxPixels)
            throw new InvalidDataException($"Image must contain between 1 and {maxPixels / 1_000_000} million pixels: {label}");
        var rgb = new byte[checked(width * height * 3)];
        if (!ImageCodecs.TryDecodeInto(data, rgb, width * 3, PixelFormat.Rgb24, options, cancellationToken))
            throw new InvalidDataException($"Could not decode image: {label}");
        return new RgbImage(rgb, width, height);
    }
}

public readonly record struct RgbImage(byte[] Rgb, int Width, int Height);
