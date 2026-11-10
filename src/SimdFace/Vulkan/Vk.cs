// Vulkan ABI declarations adapted from SimdSkia (MIT). See SimdSkia.LICENSE.txt.
using System.Reflection;
using System.Runtime.InteropServices;

namespace SimdFace.Vulkan;

internal static unsafe class Vk
{
    private const string Library = "SimdFace.Vulkan.Loader";

    static Vk()
    {
        if (IntPtr.Size != 8 || !BitConverter.IsLittleEndian || !(OperatingSystem.IsWindows() || OperatingSystem.IsLinux()))
            throw new PlatformNotSupportedException("The Vulkan backend supports 64-bit Windows and Linux.");
        NativeLibrary.SetDllImportResolver(typeof(Vk).Assembly, Resolve);
    }

    private static nint Resolve(string name, Assembly assembly, DllImportSearchPath? search)
    {
        if (name != Library)
            return 0;
        return OperatingSystem.IsWindows()
            ? NativeLibrary.Load("vulkan-1.dll", assembly, DllImportSearchPath.System32)
            : NativeLibrary.Load("libvulkan.so.1", assembly, search);
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct InstanceInfo
    {
        public uint Type;
        public void* Next;
        public uint Flags;
        public void* Application;
        public uint LayerCount;
        public void* Layers;
        public uint ExtensionCount;
        public void* Extensions;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct QueueInfo
    {
        public uint Type;
        public void* Next;
        public uint Flags,Family,Count;
        public float* Priorities;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct DeviceInfo
    {
        public uint Type;
        public void* Next;
        public uint Flags,QueueCount;
        public QueueInfo* Queues;
        public uint LayerCount;
        public void* Layers;
        public uint ExtensionCount;
        public void* Extensions;
        public void* Features;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct QueueFamily
    {
        public uint Flags,Count,TimestampBits,X,Y,Z;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct MemoryProperties
    {
        public uint TypeCount;
        public fixed uint Types[64];
        public uint HeapCount;
        public fixed ulong Heaps[32];
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct BufferInfo
    {
        public uint Type;
        public void* Next;
        public uint Flags;
        public ulong Size;
        public uint Usage,Sharing,FamilyCount;
        public uint* Families;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct MemoryRequirements
    {
        public ulong Size,Alignment;
        public uint TypeBits;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct AllocateInfo
    {
        public uint Type;
        public void* Next;
        public ulong Size;
        public uint MemoryType;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ShaderInfo
    {
        public uint Type;
        public void* Next;
        public uint Flags;
        public nuint CodeSize;
        public uint* Code;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ShaderStage
    {
        public uint Type;
        public void* Next;
        public uint Flags,Stage;
        public ulong Module;
        public byte* Name;
        public void* Specialization;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct PipelineLayoutInfo
    {
        public uint Type;
        public void* Next;
        public uint Flags,Count;
        public ulong* Layouts;
        public uint PushCount;
        public void* PushRanges;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ComputePipelineInfo
    {
        public uint Type;
        public void* Next;
        public uint Flags;
        public ShaderStage Stage;
        public ulong Layout,Base;
        public int BaseIndex;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct LayoutBinding
    {
        public uint Binding,DescriptorType,Count,Stages;
        public void* Samplers;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct DescriptorLayoutInfo
    {
        public uint Type;
        public void* Next;
        public uint Flags,Count;
        public LayoutBinding* Bindings;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct PoolSize
    {
        public uint Type,Count;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct PoolInfo
    {
        public uint Type;
        public void* Next;
        public uint Flags,MaxSets,Count;
        public PoolSize* Sizes;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SetInfo
    {
        public uint Type;
        public void* Next;
        public ulong Pool;
        public uint Count;
        public ulong* Layouts;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct DescriptorBuffer
    {
        public ulong Buffer,Offset,Range;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct WriteDescriptor
    {
        public uint Type;
        public void* Next;
        public ulong Set;
        public uint Binding,Element,Count,DescriptorType;
        public void* Image;
        public DescriptorBuffer* Buffer;
        public void* Texel;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CommandPoolInfo
    {
        public uint Type;
        public void* Next;
        public uint Flags,Family;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CommandAllocateInfo
    {
        public uint Type;
        public void* Next;
        public ulong Pool;
        public uint Level,Count;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CommandBeginInfo
    {
        public uint Type;
        public void* Next;
        public uint Flags;
        public void* Inheritance;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct FenceInfo
    {
        public uint Type;
        public void* Next;
        public uint Flags;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SubmitInfo
    {
        public uint Type;
        public void* Next;
        public uint WaitCount;
        public void* Waits;
        public void* Stages;
        public uint CommandCount;
        public IntPtr* Commands;
        public uint SignalCount;
        public void* Signals;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct BufferBarrier
    {
        public uint Type;
        public void* Next;
        public uint SourceAccess,DestinationAccess,SourceFamily,DestinationFamily;
        public ulong Buffer,Offset,Size;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct BufferCopy
    {
        public ulong SourceOffset,DestinationOffset,Size;
    }

    public struct PushConstantRange { public uint Stages, Offset, Size; }
    public struct MemoryBarrier { public uint Type; public void* Next; public uint SourceAccess, DestinationAccess; }

    [DllImport(Library)]
    public static extern int vkCreateInstance(InstanceInfo* info, void* allocator, IntPtr* instance);
    [DllImport(Library)]
    public static extern void vkDestroyInstance(IntPtr instance, void* allocator);
    [DllImport(Library)]
    public static extern int vkEnumeratePhysicalDevices(IntPtr instance, uint* count, IntPtr* devices);
    [DllImport(Library)]
    public static extern void vkGetPhysicalDeviceProperties(IntPtr physical, byte* properties);
    [DllImport(Library)]
    public static extern void vkGetPhysicalDeviceQueueFamilyProperties(IntPtr physical, uint* count, QueueFamily* queues);
    [DllImport(Library)]
    public static extern void vkGetPhysicalDeviceMemoryProperties(IntPtr physical, MemoryProperties* properties);
    [DllImport(Library)]
    public static extern int vkCreateDevice(IntPtr physical, DeviceInfo* info, void* allocator, IntPtr* device);
    [DllImport(Library)]
    public static extern void vkDestroyDevice(IntPtr device, void* allocator);
    [DllImport(Library)]
    public static extern void vkGetDeviceQueue(IntPtr device, uint family, uint index, IntPtr* queue);
    [DllImport(Library)]
    public static extern int vkCreateBuffer(IntPtr device, BufferInfo* info, void* allocator, ulong* buffer);
    [DllImport(Library)]
    public static extern void vkDestroyBuffer(IntPtr device, ulong buffer, void* allocator);
    [DllImport(Library)]
    public static extern void vkGetBufferMemoryRequirements(IntPtr device, ulong buffer, MemoryRequirements* requirements);
    [DllImport(Library)]
    public static extern int vkAllocateMemory(IntPtr device, AllocateInfo* info, void* allocator, ulong* memory);
    [DllImport(Library)]
    public static extern void vkFreeMemory(IntPtr device, ulong memory, void* allocator);
    [DllImport(Library)]
    public static extern int vkBindBufferMemory(IntPtr device, ulong buffer, ulong memory, ulong offset);
    [DllImport(Library)]
    public static extern int vkMapMemory(IntPtr device, ulong memory, ulong offset, ulong size, uint flags, void** data);
    [DllImport(Library)]
    public static extern void vkUnmapMemory(IntPtr device, ulong memory);
    [DllImport(Library)]
    public static extern int vkCreateShaderModule(IntPtr device, ShaderInfo* info, void* allocator, ulong* module);
    [DllImport(Library)]
    public static extern void vkDestroyShaderModule(IntPtr device, ulong module, void* allocator);
    [DllImport(Library)]
    public static extern int vkCreateDescriptorSetLayout(IntPtr device, DescriptorLayoutInfo* info, void* allocator, ulong* layout);
    [DllImport(Library)]
    public static extern void vkDestroyDescriptorSetLayout(IntPtr device, ulong layout, void* allocator);
    [DllImport(Library)]
    public static extern int vkCreatePipelineLayout(IntPtr device, PipelineLayoutInfo* info, void* allocator, ulong* layout);
    [DllImport(Library)]
    public static extern void vkDestroyPipelineLayout(IntPtr device, ulong layout, void* allocator);
    [DllImport(Library)]
    public static extern int vkCreateComputePipelines(IntPtr device, ulong cache, uint count, ComputePipelineInfo* infos, void* allocator, ulong* pipeline);
    [DllImport(Library)]
    public static extern void vkDestroyPipeline(IntPtr device, ulong pipeline, void* allocator);
    [DllImport(Library)]
    public static extern int vkCreateDescriptorPool(IntPtr device, PoolInfo* info, void* allocator, ulong* pool);
    [DllImport(Library)]
    public static extern void vkDestroyDescriptorPool(IntPtr device, ulong pool, void* allocator);
    [DllImport(Library)]
    public static extern int vkAllocateDescriptorSets(IntPtr device, SetInfo* info, ulong* sets);
    [DllImport(Library)]
    public static extern void vkUpdateDescriptorSets(IntPtr device, uint count, WriteDescriptor* writes, uint copies, void* copyInfo);
    [DllImport(Library)]
    public static extern int vkCreateCommandPool(IntPtr device, CommandPoolInfo* info, void* allocator, ulong* pool);
    [DllImport(Library)]
    public static extern void vkDestroyCommandPool(IntPtr device, ulong pool, void* allocator);
    [DllImport(Library)]
    public static extern int vkResetCommandPool(IntPtr device, ulong pool, uint flags);
    [DllImport(Library)]
    public static extern int vkAllocateCommandBuffers(IntPtr device, CommandAllocateInfo* info, IntPtr* buffer);
    [DllImport(Library)]
    public static extern int vkBeginCommandBuffer(IntPtr buffer, CommandBeginInfo* info);
    [DllImport(Library)]
    public static extern int vkEndCommandBuffer(IntPtr buffer);
    [DllImport(Library)]
    public static extern void vkCmdBindPipeline(IntPtr command, uint point, ulong pipeline);
    [DllImport(Library)]
    public static extern void vkCmdBindDescriptorSets(IntPtr command, uint point, ulong layout, uint first, uint count, ulong* sets, uint dynamicCount, uint* offsets);
    [DllImport(Library)]
    public static extern void vkCmdCopyBuffer(IntPtr command, ulong source, ulong destination, uint count, BufferCopy* regions);
    [DllImport(Library)]
    public static extern void vkCmdDispatch(IntPtr command, uint x, uint y, uint z);
    [DllImport(Library)]
    public static extern void vkCmdPipelineBarrier(IntPtr command, uint source, uint destination, uint dependencies, uint memoryCount, void* memories, uint bufferCount, BufferBarrier* buffers, uint imageCount, void* images);
    [DllImport(Library)]
    public static extern int vkCreateFence(IntPtr device, FenceInfo* info, void* allocator, ulong* fence);
    [DllImport(Library)]
    public static extern void vkDestroyFence(IntPtr device, ulong fence, void* allocator);
    [DllImport(Library)]
    public static extern int vkResetFences(IntPtr device, uint count, ulong* fences);
    [DllImport(Library)]
    public static extern int vkWaitForFences(IntPtr device, uint count, ulong* fences, uint all, ulong timeout);
    [DllImport(Library)]
    public static extern int vkQueueSubmit(IntPtr queue, uint count, SubmitInfo* info, ulong fence);
    [DllImport(Library)]
    public static extern int vkDeviceWaitIdle(IntPtr device);
    [DllImport(Library)]
    public static extern void vkCmdPushConstants(nint command, ulong layout, uint stages, uint offset, uint size, void* data);
}
