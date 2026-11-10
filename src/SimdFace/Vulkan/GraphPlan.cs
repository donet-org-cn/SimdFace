namespace SimdFace;

internal sealed partial class ModelGraph
{
    internal GraphPlan PlanVulkan(Tensor input)
    {
        int next = 0;
        var constants = new Dictionary<string, int>(StringComparer.Ordinal);
        var uploads = new List<float[]>();
        foreach (var (name, weight) in weights)
        {
            constants.Add(name, next);
            uploads.Add(weight.Data);
            next = checked(next + weight.Data.Length);
        }

        int weightLength = next;
        var inputSlot = new TensorSlot(next, input.Channels, input.Height, input.Width);
        next = checked(next + inputSlot.Length);
        var slots = new Dictionary<string, TensorSlot>(StringComparer.Ordinal) { [inputName] = inputSlot };
        var dispatches = new List<Dispatch>(layers.Count);

        foreach (Layer layer in layers)
        {
            string[] args = layer.Inputs;
            TensorSlot src = slots[args[0]];
            if (layer.Op is "Flatten" or "Reshape" or "Dropout")
            {
                slots.Add(layer.Output, src);
                continue;
            }

            int channels = src.Channels, height = src.Height, width = src.Width;
            var p = new Dispatch
            {
                Source = src.Offset, Destination = next,
                InChannels = src.Channels, InHeight = src.Height, InWidth = src.Width
            };
            switch (layer.Op)
            {
                case "Conv":
                    int[] shape = weights[args[1]].Shape;
                    channels = shape[0];
                    p.KernelHeight = shape[2];
                    p.KernelWidth = shape[3];
                    p.StrideY = layer.Strides[0];
                    p.StrideX = layer.Strides[1];
                    p.PadY = layer.Pads[0];
                    p.PadX = layer.Pads[1];
                    p.Groups = layer.Group;
                    height = (height + p.PadY + layer.Pads[2] - p.KernelHeight) / p.StrideY + 1;
                    width = (width + p.PadX + layer.Pads[3] - p.KernelWidth) / p.StrideX + 1;
                    p.Weights = constants[args[1]];
                    p.Bias = args.Length == 3 ? constants[args[2]] : -1;
                    p.Op = p.KernelHeight == 1 && p.KernelWidth == 1 && p.Groups == 1 &&
                        p.StrideY == 1 && p.StrideX == 1 && layer.Pads.All(pad => pad == 0) ? 13 : 0;
                    break;
                case "BatchNormalization":
                    p.Op = 1;
                    p.Weights = constants[args[1]];
                    p.Bias = constants[args[2]];
                    p.Auxiliary = constants[args[3]];
                    p.Extra = constants[args[4]];
                    p.Scalar = layer.Epsilon;
                    break;
                case "PRelu": p.Op = 2; p.Weights = constants[args[1]]; break;
                case "Relu": p.Op = 3; break;
                case "MaxPool": p.Op = 4; height /= 2; width /= 2; break;
                case "Resize": p.Op = 5; height *= 2; width *= 2; break;
                case "Transpose": p.Op = 6; break;
                case "Sigmoid": p.Op = 7; break;
                case "Add": p.Op = 8; p.Auxiliary = slots[args[1]].Offset; break;
                case "Sub": p.Op = 9; p.Scalar = weights[args[1]].Data[0]; break;
                case "Mul": p.Op = 10; p.Scalar = weights[args[1]].Data[0]; break;
                case "Gemm":
                    p.Op = 12;
                    p.Weights = constants[args[1]];
                    p.Bias = constants[args[2]];
                    channels = weights[args[1]].Shape[0];
                    height = width = 1;
                    break;
                default: throw new NotSupportedException($"Unsupported Vulkan model operator: {layer.Op}");
            }

            p.OutChannels = channels;
            p.OutHeight = height;
            p.OutWidth = width;
            var output = new TensorSlot(next, channels, height, width);
            slots.Add(layer.Output, output);
            next = checked(next + output.Length);
            dispatches.Add(p);
        }

        return new GraphPlan(inputSlot, outputNames.Select(name => slots[name]).ToArray(),
            dispatches.ToArray(), uploads.ToArray(), weightLength, next);
    }
}

internal readonly record struct TensorSlot(int Offset, int Channels, int Height, int Width)
{
    public int Length => checked(Channels * Height * Width);
}

internal sealed record GraphPlan(TensorSlot Input, TensorSlot[] Outputs, Dispatch[] Dispatches,
    float[][] Weights, int WeightLength, int Length);

// This layout matches the push constants in both GLSL shaders.
internal struct Dispatch
{
    public int Op, Source, Destination, Weights, Bias, Auxiliary, Extra;
    public int InChannels, InHeight, InWidth, OutChannels, OutHeight, OutWidth;
    public int KernelHeight, KernelWidth, StrideY, StrideX, PadY, PadX, Groups;
    public float Scalar;
}
