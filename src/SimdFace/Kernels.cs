using System.Buffers;
using System.Numerics;

namespace SimdFace;

internal static class Kernels
{
#if !NETWASM
    private static readonly ParallelOptions Cpu = new() { MaxDegreeOfParallelism = Math.Min(8, Environment.ProcessorCount) };
#endif

    private static void ForChannels(int count, Action<int> body)
    {
#if NETWASM
        for (int i = 0; i < count; i++)
            body(i);
#else
        Parallel.For(0, count, Cpu, body);
#endif
    }

    public static Tensor Conv(Tensor input, Weights filter, float[]? bias, int sy, int sx, int[] pad, int groups)
    {
        int channels = filter.Shape[0], perGroup = filter.Shape[1];
        int kh = filter.Shape[2], kw = filter.Shape[3];
        int height = (input.Height + pad[0] + pad[2] - kh) / sy + 1;
        int width = (input.Width + pad[1] + pad[3] - kw) / sx + 1;
        var output = new Tensor(channels, height, width, new float[channels * height * width]);
        bool depthwise = groups == input.Channels && channels == groups && groups > 1;
        if (depthwise)
        {
            ForChannels(channels, c => Depthwise(input, output, filter.Data, bias, c, kh, kw, sy, sx, pad));
            return output;
        }

        int terms = perGroup * kh * kw;
        bool pointwise = kh == 1 && kw == 1 && sy == 1 && sx == 1 && pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0;
        float[] columns = pointwise ? input.Data : ArrayPool<float>.Shared.Rent(input.Channels * kh * kw * output.Plane);
        try
        {
            if (!pointwise)
                FillImageColumns(input, output, kh, kw, sy, sx, pad, columns);
            ForChannels(channels, c =>
            {
                Span<float> row = output.Data.AsSpan(c * output.Plane, output.Plane);
                row.Fill(bias?[c] ?? 0);
                int inputStart = c / (channels / groups) * terms * output.Plane;
                for (int k = 0; k < terms; k++)
                {
                    float scale = filter.Data[c * terms + k];
                    Accumulate(row, columns.AsSpan(inputStart + k * output.Plane, output.Plane), scale);
                }
            });
            return output;
        }
        finally
        {
            if (!pointwise)
                ArrayPool<float>.Shared.Return(columns);
        }
    }

    private static void FillImageColumns(Tensor input, Tensor output, int kh, int kw, int sy, int sx, int[] pad,
        float[] columns)
    {
        ForChannels(input.Channels, c =>
        {
            for (int ky = 0; ky < kh; ky++)
            for (int kx = 0; kx < kw; kx++)
            {
                int offset = (c * kh * kw + ky * kw + kx) * output.Plane;
                int dx = kx - pad[1];
                int first = Math.Clamp((-dx + sx - 1) / sx, 0, output.Width);
                int end = Math.Clamp((input.Width - dx + sx - 1) / sx, first, output.Width);
                for (int y = 0; y < output.Height; y++)
                {
                    var row = columns.AsSpan(offset + y * output.Width, output.Width);
                    int iy = y * sy + ky - pad[0];
                    if ((uint)iy >= (uint)input.Height)
                    {
                        row.Clear();
                        continue;
                    }
                    int source = c * input.Plane + iy * input.Width + dx;
                    // Rented storage can contain data from a prior inference.
                    row[..first].Clear();
                    for (int x = first; x < end; x++)
                        row[x] = input.Data[source + x * sx];
                    row[end..].Clear();
                }
            }
        });
    }

    private static void Depthwise(Tensor input, Tensor output, float[] filter, float[]? bias, int channel,
        int kh, int kw, int sy, int sx, int[] pad)
    {
        var plane = output.Data.AsSpan(channel * output.Plane, output.Plane);
        plane.Fill(bias?[channel] ?? 0);
        int inputOffset = channel * input.Plane;
        for (int ky = 0; ky < kh; ky++)
        for (int kx = 0; kx < kw; kx++)
        {
            float scale = filter[(channel * kh + ky) * kw + kx];
            int dx = kx - pad[1];
            int first = Math.Max(0, (-dx + sx - 1) / sx);
            int end = Math.Clamp((input.Width - dx + sx - 1) / sx, first, output.Width);
            for (int y = 0; y < output.Height; y++)
            {
                int iy = y * sy + ky - pad[0];
                if ((uint)iy >= (uint)input.Height || first == end)
                    continue;
                int source = inputOffset + iy * input.Width;
                var row = plane.Slice(y * output.Width, output.Width);
                if (sx == 1)
                    Accumulate(row[first..end], input.Data.AsSpan(source + first + dx, end - first), scale);
                else
                    for (int x = first; x < end; x++)
                        row[x] += input.Data[source + x * sx + dx] * scale;
            }
        }
    }

    private static void Accumulate(Span<float> destination, ReadOnlySpan<float> source, float scale)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var factor = new Vector<float>(scale);
            for (; i <= destination.Length - Vector<float>.Count; i += Vector<float>.Count)
                (new Vector<float>(destination[i..]) + new Vector<float>(source[i..]) * factor).CopyTo(destination[i..]);
        }
        for (; i < destination.Length; i++)
            destination[i] += source[i] * scale;
    }

    public static void NormalizeAndActivate(Tensor x, float[] scale, float[] offset, float[] slope)
    {
        for (int c = 0; c < x.Channels; c++)
        {
            var plane = x.Data.AsSpan(c * x.Plane, x.Plane);
            int i = 0;
            if (Vector.IsHardwareAccelerated)
            {
                var factor = new Vector<float>(scale[c]);
                var bias = new Vector<float>(offset[c]);
                var negativeScale = new Vector<float>(slope[c]);
                for (; i <= plane.Length - Vector<float>.Count; i += Vector<float>.Count)
                {
                    var value = new Vector<float>(plane[i..]) * factor + bias;
                    Vector.ConditionalSelect(Vector.GreaterThanOrEqual(value, Vector<float>.Zero),
                        value, value * negativeScale).CopyTo(plane[i..]);
                }
            }
            for (; i < plane.Length; i++)
            {
                float value = plane[i] * scale[c] + offset[c];
                plane[i] = value >= 0 ? value : value * slope[c];
            }
        }
    }

    public static Tensor BatchNorm(Tensor x, float[] gamma, float[] beta, float[] mean, float[] variance, float epsilon)
    {
        var result = new float[x.Data.Length];
        for (int c = 0; c < x.Channels; c++)
        {
            float scale = gamma[c] / MathF.Sqrt(variance[c] + epsilon);
            float offset = beta[c] - mean[c] * scale;
            int end = (c + 1) * x.Plane;
            for (int i = c * x.Plane; i < end; i++)
                result[i] = x.Data[i] * scale + offset;
        }
        return x with { Data = result };
    }

    public static void ReluInPlace(Tensor x)
    {
        var values = x.Data.AsSpan();
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var zero = Vector<float>.Zero;
            for (; i <= values.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                var value = new Vector<float>(values[i..]);
                Vector.ConditionalSelect(Vector.GreaterThanOrEqual(value, zero), value, value * zero).CopyTo(values[i..]);
            }
        }
        for (; i < values.Length; i++)
        {
            float value = values[i];
            values[i] = value >= 0 ? value : value * 0;
        }
    }

    public static Tensor PRelu(Tensor x, float[]? slope)
    {
        var result = new float[x.Data.Length];
        for (int c = 0; c < x.Channels; c++)
        {
            float negativeScale = slope?[c] ?? 0;
            int end = (c + 1) * x.Plane;
            for (int i = c * x.Plane; i < end; i++)
                result[i] = x.Data[i] >= 0 ? x.Data[i] : x.Data[i] * negativeScale;
        }
        return x with { Data = result };
    }

    public static Tensor Pool2(Tensor x)
    {
        var output = new Tensor(x.Channels, x.Height / 2, x.Width / 2, new float[x.Channels * (x.Height / 2) * (x.Width / 2)]);
        for (int c = 0; c < x.Channels; c++)
        for (int y = 0; y < output.Height; y++)
        for (int col = 0; col < output.Width; col++)
        {
            int i = c * x.Plane + 2 * y * x.Width + 2 * col;
            output.Data[c * output.Plane + y * output.Width + col] = MathF.Max(
                MathF.Max(x.Data[i], x.Data[i + 1]), MathF.Max(x.Data[i + x.Width], x.Data[i + x.Width + 1]));
        }
        return output;
    }

    public static Tensor Upsample2(Tensor x)
    {
        var output = new Tensor(x.Channels, x.Height * 2, x.Width * 2, new float[x.Data.Length * 4]);
        for (int c = 0; c < x.Channels; c++)
        for (int y = 0; y < output.Height; y++)
        for (int col = 0; col < output.Width; col++)
            output.Data[c * output.Plane + y * output.Width + col] = x.Data[c * x.Plane + (y / 2) * x.Width + col / 2];
        return output;
    }

    public static Tensor ToInterleaved(Tensor x)
    {
        var output = new float[x.Data.Length];
        for (int c = 0; c < x.Channels; c++)
        for (int i = 0; i < x.Plane; i++)
            output[i * x.Channels + c] = x.Data[c * x.Plane + i];
        return x with { Data = output };
    }

    public static Tensor Map(Tensor x, Converter<float, float> map) => x with { Data = Array.ConvertAll(x.Data, map) };

    public static Tensor Add(Tensor x, Tensor y)
    {
        var output = new float[x.Data.Length];
        for (int i = 0; i < output.Length; i++)
            output[i] = x.Data[i] + y.Data[i];
        return x with { Data = output };
    }

    public static Tensor Dense(Tensor x, Weights matrix, float[] bias)
    {
        var output = new float[matrix.Shape[0]];
        ForChannels(output.Length, c =>
            output[c] = bias[c] + Dot(x.Data, matrix.Data.AsSpan(c * x.Data.Length, x.Data.Length)));
        return new Tensor(output.Length, 1, 1, output);
    }

    public static float Dot(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        int i = 0;
        var accumulator = Vector<float>.Zero;
        if (Vector.IsHardwareAccelerated)
            for (; i <= x.Length - Vector<float>.Count; i += Vector<float>.Count)
                accumulator += new Vector<float>(x[i..]) * new Vector<float>(y[i..]);
        float sum = Vector.Sum(accumulator);
        for (; i < x.Length; i++)
            sum += x[i] * y[i];
        return sum;
    }
}
