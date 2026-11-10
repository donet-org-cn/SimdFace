using System.Buffers.Binary;
using System.Runtime.InteropServices;
#if !NETWASM
using System.Security.Cryptography;
#endif
using System.Text;

namespace SimdFace;

// Reads only the ONNX fields and operators used by the two pinned models.
internal sealed partial class ModelGraph
{
    private readonly Dictionary<string, Weights> weights = new(StringComparer.Ordinal);
    private readonly List<Layer> layers = [];
    private readonly Dictionary<string, int> lastUse = new(StringComparer.Ordinal);
    private readonly Dictionary<int, ConvActivation?> convActivations = [];
    private readonly string inputName;
    private readonly string[] outputNames;

#if !NETWASM
    private readonly VulkanInference? vulkan;

    private const int MaxModelBytes = 40_000_000;

    internal static byte[] ReadFile(string path)
    {
        using var file = File.OpenRead(path);
        if (file.Length > MaxModelBytes)
            throw new InvalidDataException($"Unexpected model size: {path}");
        var bytes = new byte[(int)file.Length];
        file.ReadExactly(bytes);
        return bytes;
    }

    public ModelGraph(ReadOnlySpan<byte> bytes, string label, string sha256, string inputName,
        VulkanInference? vulkan, params string[] outputNames)
    {
        this.vulkan = vulkan;
        if (bytes.Length > MaxModelBytes)
            throw new InvalidDataException($"Unexpected model size: {label}");
        if (!Convert.ToHexString(SHA256.HashData(bytes)).Equals(sha256, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"Model SHA-256 mismatch: {label}. Use the version listed in the README.");
#else
    public ModelGraph(ReadOnlySpan<byte> bytes, string inputName, params string[] outputNames)
    {
#endif
        this.inputName = inputName;
        this.outputNames = outputNames;

        var model = new ProtoReader(bytes);
        while (model.Next(out int field, out _, out var value))
        {
            if (field != 7)
                continue;
            var graph = new ProtoReader(value);
            while (graph.Next(out field, out _, out value))
            {
                if (field == 1)
                    layers.Add(ReadLayer(value));
                if (field == 5)
                    ReadWeights(value);
            }
            break;
        }

        for (int i = 0; i < layers.Count; i++)
            foreach (string name in layers[i].Inputs)
                if (!weights.ContainsKey(name))
                    lastUse[name] = i;
        foreach (string name in outputNames)
            lastUse[name] = layers.Count;

#if !NETWASM
        if (vulkan is not null)
            return;
#endif
        PrepareConvActivations();
    }

    private void PrepareConvActivations()
    {
        for (int i = 0; i + 1 < layers.Count; i++)
        {
            Layer conv = layers[i], norm = layers[i + 1];
            // Only mutate a fresh convolution output with a single remaining consumer.
            if (conv.Op != "Conv" || norm.Inputs[0] != conv.Output || lastUse[conv.Output] != i + 1)
                continue;
            if (norm.Op == "Relu")
            {
                convActivations.Add(i, null);
                continue;
            }
            if (norm.Op != "BatchNormalization" || i + 2 == layers.Count)
                continue;
            Layer activation = layers[i + 2];
            if (activation.Op != "PRelu" || activation.Inputs[0] != norm.Output || lastUse[norm.Output] != i + 2)
                continue;

            float[] gamma = weights[norm.Inputs[1]].Data, beta = weights[norm.Inputs[2]].Data;
            float[] mean = weights[norm.Inputs[3]].Data, variance = weights[norm.Inputs[4]].Data;
            var scale = new float[gamma.Length];
            var offset = new float[gamma.Length];
            for (int c = 0; c < scale.Length; c++)
            {
                scale[c] = gamma[c] / MathF.Sqrt(variance[c] + norm.Epsilon);
                offset[c] = beta[c] - mean[c] * scale[c];
            }
            convActivations.Add(i, new ConvActivation(scale, offset, weights[activation.Inputs[1]].Data));
        }
    }

#if !NETWASM
    internal bool UsesVulkan => vulkan is not null;

    internal float[][] RunVulkanBatch(Tensor[] inputs) => vulkan!.RunBatch(this, inputs);
#endif

    public float[][] Run(Tensor input)
    {
#if !NETWASM
        if (vulkan is not null)
            return vulkan.Run(this, input);
#endif

        var values = new Dictionary<string, Tensor>(StringComparer.Ordinal) { [inputName] = input };
        for (int i = 0; i < layers.Count; i++)
        {
            Layer layer = layers[i];
            string[] args = layer.Inputs;
            Tensor x = values[args[0]];
            Tensor output = layer.Op switch
            {
                "Conv" => Kernels.Conv(x, weights[args[1]], args.Length == 3 ? weights[args[2]].Data : null,
                    layer.Strides[0], layer.Strides[1], layer.Pads, layer.Group),
                "BatchNormalization" => Kernels.BatchNorm(x, weights[args[1]].Data, weights[args[2]].Data,
                    weights[args[3]].Data, weights[args[4]].Data, layer.Epsilon),
                "PRelu" => Kernels.PRelu(x, weights[args[1]].Data),
                "Relu" => Kernels.PRelu(x, null),
                "MaxPool" => Kernels.Pool2(x),
                "Resize" => Kernels.Upsample2(x),
                "Transpose" => Kernels.ToInterleaved(x),
                "Sigmoid" => Kernels.Map(x, value => 1f / (1f + MathF.Exp(-value))),
                "Add" => Kernels.Add(x, values[args[1]]),
                "Sub" => Kernels.Map(x, value => value - weights[args[1]].Data[0]),
                "Mul" => Kernels.Map(x, value => value * weights[args[1]].Data[0]),
                "Gemm" => Kernels.Dense(x, weights[args[1]], weights[args[2]].Data),
                // The following layers preserve the flat data in these graphs.
                "Flatten" or "Reshape" or "Dropout" => x,
                _ => throw new NotSupportedException($"Unsupported model operator: {layer.Op}")
            };
            foreach (string name in args)
                if (lastUse.TryGetValue(name, out int last) && last == i)
                    values.Remove(name);

            if (convActivations.TryGetValue(i, out var activation))
            {
                if (activation is null)
                {
                    Kernels.ReluInPlace(output);
                    i++;
                }
                else
                {
                    Kernels.NormalizeAndActivate(output, activation.Scale, activation.Offset, activation.Slope);
                    i += 2;
                }
            }
            values[layers[i].Output] = output;
        }
        return Array.ConvertAll(outputNames, name => values[name].Data);
    }

    private void ReadWeights(ReadOnlySpan<byte> data)
    {
        var reader = new ProtoReader(data);
        var shape = new List<int>(4);
        var scalarValues = new List<float>();
        string name = "";
        int type = 0;
        ReadOnlySpan<byte> raw = default;
        while (reader.Next(out int field, out ulong number, out var value))
        {
            switch (field)
            {
                case 1: ReadInts(shape, number, value); break;
                case 2: type = (int)number; break;
                case 4 when value.Length == 4:
                    scalarValues.Add(BinaryPrimitives.ReadSingleLittleEndian(value));
                    break;
                case 4: raw = value; break;
                case 8: name = Encoding.UTF8.GetString(value); break;
                case 9: raw = value; break;
            }
        }
        if (type == 1)
            weights.Add(name, new Weights(shape.ToArray(), raw.IsEmpty
                ? scalarValues.ToArray()
                : MemoryMarshal.Cast<byte, float>(raw).ToArray()));
    }

    private static Layer ReadLayer(ReadOnlySpan<byte> data)
    {
        var reader = new ProtoReader(data);
        var layer = new Layer();
        var inputs = new List<string>();
        while (reader.Next(out int field, out _, out var value))
        {
            switch (field)
            {
                case 1: inputs.Add(Encoding.UTF8.GetString(value)); break;
                case 2: layer.Output = Encoding.UTF8.GetString(value); break;
                case 4: layer.Op = Encoding.UTF8.GetString(value); break;
                case 5: ReadAttribute(layer, value); break;
            }
        }
        layer.Inputs = inputs.ToArray();
        return layer;
    }

    private static void ReadAttribute(Layer layer, ReadOnlySpan<byte> data)
    {
        var reader = new ProtoReader(data);
        string name = "";
        int integer = 0;
        float scalar = 0;
        var integers = new List<int>();
        while (reader.Next(out int field, out ulong number, out var value))
        {
            switch (field)
            {
                case 1: name = Encoding.UTF8.GetString(value); break;
                case 2: scalar = BinaryPrimitives.ReadSingleLittleEndian(value); break;
                case 3: integer = (int)number; break;
                case 8: ReadInts(integers, number, value); break;
            }
        }
        switch (name)
        {
            case "group": layer.Group = integer; break;
            case "strides": layer.Strides = integers.ToArray(); break;
            case "pads": layer.Pads = integers.ToArray(); break;
            case "epsilon": layer.Epsilon = scalar; break;
        }
    }

    private static void ReadInts(List<int> result, ulong number, ReadOnlySpan<byte> packed)
    {
        if (packed.IsEmpty)
            result.Add((int)number);
        else
        {
            var reader = new ProtoReader(packed);
            while (!reader.AtEnd)
                result.Add((int)reader.ReadVarint());
        }
    }

    private sealed record ConvActivation(float[] Scale, float[] Offset, float[] Slope);

    private sealed class Layer
    {
        public string Op = "";
        public string Output = "";
        public string[] Inputs = [];
        public int Group = 1;
        public int[] Strides = [1, 1];
        public int[] Pads = [0, 0, 0, 0];
        public float Epsilon = 1e-5f;
    }
}

internal sealed record Weights(int[] Shape, float[] Data);

internal sealed record Tensor(int Channels, int Height, int Width, float[] Data)
{
    public int Plane => Height * Width;
}

internal ref struct ProtoReader
{
    private readonly ReadOnlySpan<byte> data;
    private int position;

    public ProtoReader(ReadOnlySpan<byte> data) => this.data = data;

    public readonly bool AtEnd => position == data.Length;

    public bool Next(out int field, out ulong number, out ReadOnlySpan<byte> value)
    {
        field = 0;
        number = 0;
        value = default;
        if (AtEnd)
            return false;
        ulong tag = ReadVarint();
        field = (int)(tag >> 3);
        switch (tag & 7)
        {
            case 0: number = ReadVarint(); break;
            case 1: value = Take(8); break;
            case 2: value = Take(checked((int)ReadVarint())); break;
            case 5: value = Take(4); break;
            default: throw new InvalidDataException("Unsupported ONNX protobuf wire type.");
        }
        return true;
    }

    public ulong ReadVarint()
    {
        ulong result = 0;
        for (int shift = 0; shift < 64 && position < data.Length; shift += 7)
        {
            byte next = data[position++];
            result |= (ulong)(next & 0x7f) << shift;
            if (next < 0x80)
                return result;
        }
        throw new InvalidDataException("Invalid ONNX protobuf varint.");
    }

    private ReadOnlySpan<byte> Take(int length)
    {
        if (length > data.Length - position)
            throw new InvalidDataException("Truncated ONNX protobuf field.");
        var value = data.Slice(position, length);
        position += length;
        return value;
    }
}
