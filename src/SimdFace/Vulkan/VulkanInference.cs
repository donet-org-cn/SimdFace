using System.Text;
using SimdFace.Vulkan;

namespace SimdFace;

/// <summary>A reusable Vulkan device for YuNet and SFace inference. Dispose it after both models finish.</summary>
public sealed unsafe class VulkanInference : IDisposable
{
    private const uint ComputeShader = 0x20;
    private const uint ComputeStage = 0x800;
    private const uint TransferStage = 0x1000;
    private const uint HostStage = 0x4000;
    private const uint ShaderRead = 0x20, ShaderWrite = 0x40;
    private const uint TransferRead = 0x800, TransferWrite = 0x1000;
    private const uint HostRead = 0x2000;
    private const uint StorageDescriptor = 7;
    private const uint ComputeBindPoint = 1;
    private const uint TransferSource = 1, TransferDestination = 2, StorageBuffer = 0x20;
    private const uint DeviceLocal = 1, HostCoherentVisible = 6;

    private readonly object gate = new();
    private readonly Dictionary<ModelGraph, ModelState> models = new();
    private nint instance, device, queue, command;
    private ulong commandPool, fence, descriptorLayout, pipelineLayout, tensorPipeline, pointwisePipeline;
    private Vk.MemoryProperties memoryProperties;
    private bool disposed;

    public string DeviceName { get; } = "";

    public VulkanInference()
    {
        try
        {
            if (sizeof(Dispatch) != 84 || sizeof(Vk.DeviceInfo) != 72 || sizeof(Vk.MemoryProperties) != 520)
                throw new PlatformNotSupportedException("Unexpected Vulkan ABI layout.");

            var instanceInfo = new Vk.InstanceInfo { Type = 1 };
            nint handle;
            Check(Vk.vkCreateInstance(&instanceInfo, null, &handle), "create instance");
            instance = handle;
            var (physical, family, name) = PickDevice();
            DeviceName = name;
            Vk.MemoryProperties properties;
            Vk.vkGetPhysicalDeviceMemoryProperties(physical, &properties);
            memoryProperties = properties;

            float priority = 1;
            var queueInfo = new Vk.QueueInfo { Type = 2, Family = family, Count = 1, Priorities = &priority };
            var deviceInfo = new Vk.DeviceInfo { Type = 3, QueueCount = 1, Queues = &queueInfo };
            Check(Vk.vkCreateDevice(physical, &deviceInfo, null, &handle), "create device");
            device = handle;
            Vk.vkGetDeviceQueue(device, family, 0, &handle);
            queue = handle;

            var poolInfo = new Vk.CommandPoolInfo { Type = 39, Family = family };
            ulong pool;
            Check(Vk.vkCreateCommandPool(device, &poolInfo, null, &pool), "create command pool");
            commandPool = pool;
            var allocate = new Vk.CommandAllocateInfo { Type = 40, Pool = commandPool, Count = 1 };
            Check(Vk.vkAllocateCommandBuffers(device, &allocate, &handle), "allocate command buffer");
            command = handle;
            var fenceInfo = new Vk.FenceInfo { Type = 8 };
            ulong completion;
            Check(Vk.vkCreateFence(device, &fenceInfo, null, &completion), "create fence");
            fence = completion;

            CreateLayouts();
            tensorPipeline = CreatePipeline("tensor");
            pointwisePipeline = CreatePipeline("pointwise");
        }
        catch (Exception error) when (error is DllNotFoundException or EntryPointNotFoundException or TypeInitializationException)
        {
            Dispose();
            throw new NotSupportedException("The system Vulkan loader is unavailable.", error);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    private (nint Physical, uint Family, string Name) PickDevice()
    {
        uint count = 0;
        Check(Vk.vkEnumeratePhysicalDevices(instance, &count, null), "enumerate devices");
        var devices = new nint[count];
        fixed (nint* list = devices)
            Check(Vk.vkEnumeratePhysicalDevices(instance, &count, list), "enumerate devices");

        (nint physical, uint family, string name) best = (0, 0, "");
        int bestRank = 0;
        byte* properties = stackalloc byte[1024];
        foreach (nint physical in devices)
        {
            Vk.vkGetPhysicalDeviceProperties(physical, properties);
            uint kind = *(uint*)(properties + 16);
            int rank = kind switch { 2 => 3, 1 => 2, 4 => 0, _ => 1 };
            if (rank <= bestRank)
                continue;

            uint familyCount = 0;
            Vk.vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, null);
            var families = new Vk.QueueFamily[familyCount];
            fixed (Vk.QueueFamily* list = families)
                Vk.vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, list);
            for (uint i = 0; i < familyCount; i++)
            {
                if (families[i].Count == 0 || (families[i].Flags & 2) == 0)
                    continue;
                var bytes = new ReadOnlySpan<byte>(properties + 20, 256);
                best = (physical, i, Encoding.UTF8.GetString(bytes[..bytes.IndexOf((byte)0)]));
                bestRank = rank;
                break;
            }
        }
        if (best.physical == 0)
            throw new NotSupportedException("No hardware Vulkan device with a compute queue was found.");
        return best;
    }

    private void CreateLayouts()
    {
        var binding = new Vk.LayoutBinding { DescriptorType = StorageDescriptor, Count = 1, Stages = ComputeShader };
        var descriptorInfo = new Vk.DescriptorLayoutInfo { Type = 32, Count = 1, Bindings = &binding };
        ulong layout;
        Check(Vk.vkCreateDescriptorSetLayout(device, &descriptorInfo, null, &layout), "create descriptor layout");
        descriptorLayout = layout;

        var push = new Vk.PushConstantRange { Stages = ComputeShader, Size = (uint)sizeof(Dispatch) };
        var pipelineInfo = new Vk.PipelineLayoutInfo { Type = 30, Count = 1, Layouts = &layout, PushCount = 1, PushRanges = &push };
        ulong result;
        Check(Vk.vkCreatePipelineLayout(device, &pipelineInfo, null, &result), "create pipeline layout");
        pipelineLayout = result;
    }

    private ulong CreatePipeline(string name)
    {
        using Stream shader = typeof(VulkanInference).Assembly.GetManifestResourceStream($"SimdFace.Vulkan.Shaders.{name}.spv")
            ?? throw new InvalidOperationException($"Missing embedded Vulkan shader {name}.spv.");
        var bytes = new byte[checked((int)shader.Length)];
        shader.ReadExactly(bytes);
        ulong module;
        fixed (byte* code = bytes)
        {
            var info = new Vk.ShaderInfo { Type = 16, CodeSize = (nuint)bytes.Length, Code = (uint*)code };
            Check(Vk.vkCreateShaderModule(device, &info, null, &module), "create shader module");
        }
        try
        {
            byte* entry = stackalloc byte[] { (byte)'m', (byte)'a', (byte)'i', (byte)'n', 0 };
            var stage = new Vk.ShaderStage { Type = 18, Stage = ComputeShader, Module = module, Name = entry };
            var info = new Vk.ComputePipelineInfo { Type = 29, Stage = stage, Layout = pipelineLayout, BaseIndex = -1 };
            ulong pipeline;
            Check(Vk.vkCreateComputePipelines(device, 0, 1, &info, null, &pipeline), $"create {name} pipeline");
            return pipeline;
        }
        finally
        {
            Vk.vkDestroyShaderModule(device, module, null);
        }
    }

    internal float[][] Run(ModelGraph graph, Tensor input) => RunBatch(graph, [input]);

    // The batch shares tensor storage while each input and output has a separate staging slice.
    internal float[][] RunBatch(ModelGraph graph, IReadOnlyList<Tensor> inputs)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            Tensor input = inputs[0];
            if (!models.TryGetValue(graph, out ModelState? state))
                models.Add(graph, state = new ModelState());
            GraphPlan? plan = state.Plan;
            if (plan is null || plan.Input.Channels != input.Channels ||
                plan.Input.Height != input.Height || plan.Input.Width != input.Width)
                state.Plan = plan = graph.PlanVulkan(input);

            int inputLength = input.Data.Length;
            int outputLength = plan.Outputs.Sum(output => output.Length);
            int inputTotal = checked(inputLength * inputs.Count);
            int outputStart = inputs.Count == 1 ? 0 : inputTotal;
            int stagingLength = Math.Max(plan.WeightLength,
                Math.Max(inputTotal, checked(outputStart + outputLength * inputs.Count)));
            Arena? arena = state.Arena;
            if (arena is null || arena.Storage.Size < Bytes(plan.Length) || arena.Staging.Size < Bytes(stagingLength))
            {
                state.Arena = null;
                arena?.Dispose();
                state.Arena = arena = CreateArena(plan, stagingLength);
            }

            var staging = arena.Staging.AsSpan(stagingLength);
            for (int i = 0; i < inputs.Count; i++)
            {
                Tensor sample = inputs[i];
                if (sample.Channels != input.Channels || sample.Height != input.Height ||
                    sample.Width != input.Width || sample.Data.Length != inputLength)
                    throw new ArgumentException("Vulkan batch inputs must share the same shape.", nameof(inputs));
                sample.Data.CopyTo(staging.Slice(i * inputLength, inputLength));
            }

            Begin();
            ulong set = arena.DescriptorSet;
            Vk.vkCmdBindDescriptorSets(command, ComputeBindPoint, pipelineLayout, 0, 1, &set, 0, null);
            ulong boundPipeline = 0;
            int outputOffset = outputStart;
            for (int i = 0; i < inputs.Count; i++)
            {
                Copy(arena.Staging, arena.Storage, i * inputLength, plan.Input.Offset, inputLength);
                Barrier(TransferStage, ComputeStage, TransferWrite, ShaderRead | ShaderWrite);
                foreach (Dispatch dispatch in plan.Dispatches)
                {
                    Dispatch p = dispatch;
                    ulong pipeline = p.Op == 13 ? pointwisePipeline : tensorPipeline;
                    if (pipeline != boundPipeline)
                    {
                        Vk.vkCmdBindPipeline(command, ComputeBindPoint, pipeline);
                        boundPipeline = pipeline;
                    }
                    Vk.vkCmdPushConstants(command, pipelineLayout, ComputeShader, 0, (uint)sizeof(Dispatch), &p);
                    uint plane = (uint)(p.OutHeight * p.OutWidth), channels = (uint)p.OutChannels;
                    if (p.Op == 13)
                        Vk.vkCmdDispatch(command, (plane + 15) / 16, (channels + 7) / 8, 1);
                    else
                        Vk.vkCmdDispatch(command, p.Op == 12 ? channels : (channels * plane + 63) / 64, 1, 1);
                    Barrier(ComputeStage, ComputeStage, ShaderWrite, ShaderRead | ShaderWrite);
                }

                Barrier(ComputeStage, TransferStage, ShaderWrite, TransferRead);
                foreach (TensorSlot output in plan.Outputs)
                {
                    Copy(arena.Storage, arena.Staging, output.Offset, outputOffset, output.Length);
                    outputOffset += output.Length;
                }
                if (i + 1 < inputs.Count)
                    Barrier(TransferStage, TransferStage, TransferRead, TransferWrite);
            }
            Barrier(TransferStage, HostStage, TransferWrite, HostRead);
            Submit();

            var results = new float[inputs.Count * plan.Outputs.Length][];
            outputOffset = outputStart;
            for (int i = 0; i < results.Length; i++)
            {
                int length = plan.Outputs[i % plan.Outputs.Length].Length;
                results[i] = staging.Slice(outputOffset, length).ToArray();
                outputOffset += length;
            }
            return results;
        }
    }

    private Arena CreateArena(GraphPlan plan, int stagingLength)
    {
        const ulong guaranteedStorageLimit = 128 * 1024 * 1024;
        if (Bytes(plan.Length) > guaranteedStorageLimit)
            throw new NotSupportedException("The model exceeds the Vulkan tensor storage limit.");

        var arena = new Arena(device);
        try
        {
            arena.Storage = CreateBuffer(Bytes(plan.Length), TransferSource | TransferDestination | StorageBuffer, DeviceLocal);
            arena.Staging = CreateBuffer(Bytes(stagingLength), TransferSource | TransferDestination, HostCoherentVisible);
            var size = new Vk.PoolSize { Type = StorageDescriptor, Count = 1 };
            var poolInfo = new Vk.PoolInfo { Type = 33, MaxSets = 1, Count = 1, Sizes = &size };
            ulong pool;
            Check(Vk.vkCreateDescriptorPool(device, &poolInfo, null, &pool), "create descriptor pool");
            arena.DescriptorPool = pool;
            ulong layout = descriptorLayout, set;
            var setInfo = new Vk.SetInfo { Type = 34, Pool = pool, Count = 1, Layouts = &layout };
            Check(Vk.vkAllocateDescriptorSets(device, &setInfo, &set), "allocate descriptor set");
            arena.DescriptorSet = set;
            var buffer = new Vk.DescriptorBuffer { Buffer = arena.Storage.Handle, Range = arena.Storage.Size };
            var write = new Vk.WriteDescriptor { Type = 35, Set = set, Count = 1, DescriptorType = StorageDescriptor, Buffer = &buffer };
            Vk.vkUpdateDescriptorSets(device, 1, &write, 0, null);

            Span<float> staging = arena.Staging.AsSpan(plan.WeightLength);
            int offset = 0;
            foreach (float[] weights in plan.Weights)
            {
                weights.CopyTo(staging[offset..]);
                offset += weights.Length;
            }
            Begin();
            Copy(arena.Staging, arena.Storage, 0, 0, plan.WeightLength);
            Barrier(TransferStage, ComputeStage, TransferWrite, ShaderRead);
            Submit();
            return arena;
        }
        catch
        {
            arena.Dispose();
            throw;
        }
    }

    private Buffer CreateBuffer(ulong size, uint usage, uint flags)
    {
        var result = new Buffer(device, size);
        try
        {
            var info = new Vk.BufferInfo { Type = 12, Size = size, Usage = usage };
            ulong handle, memory;
            Check(Vk.vkCreateBuffer(device, &info, null, &handle), "create buffer");
            result.Handle = handle;
            Vk.MemoryRequirements requirements;
            Vk.vkGetBufferMemoryRequirements(device, handle, &requirements);
            uint memoryType = FindMemory(requirements.TypeBits, flags);
            var allocate = new Vk.AllocateInfo { Type = 5, Size = requirements.Size, MemoryType = memoryType };
            Check(Vk.vkAllocateMemory(device, &allocate, null, &memory), "allocate buffer memory");
            result.Memory = memory;
            Check(Vk.vkBindBufferMemory(device, handle, memory, 0), "bind buffer memory");
            if (flags == HostCoherentVisible)
            {
                void* mapped;
                Check(Vk.vkMapMemory(device, memory, 0, size, 0, &mapped), "map staging memory");
                result.Mapped = (float*)mapped;
            }
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }

    private uint FindMemory(uint typeBits, uint flags)
    {
        Vk.MemoryProperties properties = memoryProperties;
        for (uint i = 0; i < properties.TypeCount; i++)
            if ((typeBits & (1u << (int)i)) != 0 && (properties.Types[2 * i] & flags) == flags)
                return i;
        throw new NotSupportedException($"No compatible Vulkan memory type for flags {flags}.");
    }

    private void Begin()
    {
        Check(Vk.vkResetCommandPool(device, commandPool, 0), "reset command pool");
        var info = new Vk.CommandBeginInfo { Type = 42, Flags = 1 };
        Check(Vk.vkBeginCommandBuffer(command, &info), "begin command buffer");
    }

    private void Submit()
    {
        Check(Vk.vkEndCommandBuffer(command), "end command buffer");
        nint recorded = command;
        ulong done = fence;
        var info = new Vk.SubmitInfo { Type = 4, CommandCount = 1, Commands = &recorded };
        Check(Vk.vkResetFences(device, 1, &done), "reset fence");
        Check(Vk.vkQueueSubmit(queue, 1, &info, done), "submit graph");
        Check(Vk.vkWaitForFences(device, 1, &done, 1, ulong.MaxValue), "wait for graph");
    }

    private void Copy(Buffer source, Buffer target, int srcOffset, int dstOffset, int length)
    {
        var region = new Vk.BufferCopy { SourceOffset = Bytes(srcOffset), DestinationOffset = Bytes(dstOffset), Size = Bytes(length) };
        Vk.vkCmdCopyBuffer(command, source.Handle, target.Handle, 1, &region);
    }

    private void Barrier(uint sourceStage, uint targetStage, uint sourceAccess, uint targetAccess)
    {
        var barrier = new Vk.MemoryBarrier { Type = 46, SourceAccess = sourceAccess, DestinationAccess = targetAccess };
        Vk.vkCmdPipelineBarrier(command, sourceStage, targetStage, 0, 1, &barrier, 0, null, 0, null);
    }

    private static ulong Bytes(int floats) => (ulong)floats * sizeof(float);

    private static void Check(int result, string operation)
    {
        if (result != 0)
            throw new InvalidOperationException($"Vulkan failed to {operation} (VkResult {result}).");
    }

    public void Dispose()
    {
        lock (gate)
        {
            if (disposed)
                return;
            disposed = true;
            if (device != 0)
            {
                Vk.vkDeviceWaitIdle(device);
                foreach (ModelState model in models.Values)
                    model.Arena?.Dispose();
                models.Clear();
                if (pointwisePipeline != 0) Vk.vkDestroyPipeline(device, pointwisePipeline, null);
                if (tensorPipeline != 0) Vk.vkDestroyPipeline(device, tensorPipeline, null);
                if (pipelineLayout != 0) Vk.vkDestroyPipelineLayout(device, pipelineLayout, null);
                if (descriptorLayout != 0) Vk.vkDestroyDescriptorSetLayout(device, descriptorLayout, null);
                if (fence != 0) Vk.vkDestroyFence(device, fence, null);
                if (commandPool != 0) Vk.vkDestroyCommandPool(device, commandPool, null);
                Vk.vkDestroyDevice(device, null);
            }
            if (instance != 0) Vk.vkDestroyInstance(instance, null);
        }
    }

    private sealed class ModelState
    {
        public GraphPlan? Plan;
        public Arena? Arena;
    }

    private sealed class Arena(nint device) : IDisposable
    {
        public Buffer Storage = null!;
        public Buffer Staging = null!;
        public ulong DescriptorPool, DescriptorSet;

        public void Dispose()
        {
            if (DescriptorPool != 0) Vk.vkDestroyDescriptorPool(device, DescriptorPool, null);
            Staging?.Dispose();
            Storage?.Dispose();
        }
    }

    private sealed class Buffer(nint device, ulong size) : IDisposable
    {
        public readonly ulong Size = size;
        public ulong Handle, Memory;
        public float* Mapped;

        public Span<float> AsSpan(int length) => new(Mapped, length);

        public void Dispose()
        {
            if (Mapped != null) Vk.vkUnmapMemory(device, Memory);
            if (Handle != 0) Vk.vkDestroyBuffer(device, Handle, null);
            if (Memory != 0) Vk.vkFreeMemory(device, Memory, null);
        }
    }
}
