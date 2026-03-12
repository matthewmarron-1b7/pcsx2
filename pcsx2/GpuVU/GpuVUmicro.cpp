// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

// --------------------------------------------------------------------------------------
//  GpuVUmicro – Vulkan compute-shader VU backend (implementation)
// --------------------------------------------------------------------------------------
//
// This file provides:
//   1. GpuVUContext  – a minimal, self-contained Vulkan compute context that is wholly
//      independent of the GS rendering thread.  It owns its own VkInstance, VkDevice,
//      compute VkQueue, command pool, and all pipeline / buffer resources.
//   2. GpuVUmicro0 / GpuVUmicro1  – BaseVUmicroCPU implementations that delegate VU
//      execution to the context via a single vkCmdDispatch call.
//   3. RegisterGpuVUBackend()  – registers the GPU descriptor with VUPluginRegistry.
//
// Fallback strategy
// -----------------
// When Vulkan initialisation fails (Vulkan runtime absent, no suitable device, …) the
// context is simply not created.  GpuVUmicro0/1 then forward every call to the built-in
// software interpreter so the emulator continues to work correctly.
//
// When the compute shader sets the `needs_cpu_fallback` flag (because it encountered an
// unimplemented instruction) the host reads the flag after the dispatch, resets VU state
// from the GPU buffer, and re-runs the micro-program on the software interpreter.

#include "GpuVU/GpuVUmicro.h"
#include "VUmicro.h"
#include "VU.h"
#include "common/Console.h"
#include "common/Error.h"

#ifdef ENABLE_VULKAN
#include "GS/Renderers/Vulkan/VKLoader.h"
#include <cstring>
#include <vector>
#include <string>
#endif

// ---------------------------------------------------------------------------
//  GpuVUState  (must match the layout in vu_execute.glsl)
// ---------------------------------------------------------------------------

struct alignas(16) GpuVUState
{
	float    VF[32][4];      // 32 × vec4
	uint32_t VI[32];         // 32 × uint (only low 16 bits are valid)
	float    ACC[4];         // accumulator vec4
	float    Q;              // quotient register
	float    P;              // EFU result register
	uint32_t TPC;            // thread program counter (byte offset)
	uint32_t cycle;          // cycle counter
	uint32_t ebit;           // E-bit countdown
	uint32_t macflag;
	uint32_t statusflag;
	uint32_t clipflag;
	uint32_t flags;          // VUFLAG_* bits
	uint32_t needs_cpu_fallback;
	uint32_t _pad[2];
};

// ---------------------------------------------------------------------------
//  GpuVUContext
// ---------------------------------------------------------------------------

class GpuVUContext
{
public:
	GpuVUContext()  = default;
	~GpuVUContext() { Destroy(); }

	bool IsReady() const { return m_ready; }

	/// Initialise the Vulkan compute context.
	bool Initialize(bool isVU1);

	/// Upload VURegs to the GPU buffers, dispatch the compute shader, wait for
	/// completion, then copy results back.  Returns false if a CPU fallback is needed.
	bool Execute(VURegs* vu, uint32_t cycles, bool isVU1);

	/// Reset internal GPU state buffers.
	void Reset(VURegs* vu, bool isVU1);

private:
	void Destroy();
	bool CreateBuffers(VkDeviceSize stateSize, VkDeviceSize microSize, VkDeviceSize memSize);
	bool CreatePipeline(const std::string& shaderGlsl);
	bool AllocateMemory(VkBuffer buf, VkDeviceMemory& mem, VkMemoryPropertyFlags props);
	void CopyVURegsToState(const VURegs* vu, bool isVU1, GpuVUState& out_state) const;
	void CopyStateToVURegs(const GpuVUState& state, VURegs* vu, bool isVU1) const;

#ifdef ENABLE_VULKAN
	bool m_ready = false;
	bool m_isVU1 = false;

	VkInstance       m_instance      = VK_NULL_HANDLE;
	VkPhysicalDevice m_phys_device   = VK_NULL_HANDLE;
	VkDevice         m_device        = VK_NULL_HANDLE;
	VkQueue          m_compute_queue = VK_NULL_HANDLE;
	uint32_t         m_queue_family  = 0;

	VkCommandPool    m_cmd_pool    = VK_NULL_HANDLE;
	VkCommandBuffer  m_cmd_buf     = VK_NULL_HANDLE;
	VkFence          m_fence       = VK_NULL_HANDLE;

	// Buffers (host-visible / coherent for simplicity; can be optimised later)
	VkBuffer       m_state_buf  = VK_NULL_HANDLE;
	VkDeviceMemory m_state_mem  = VK_NULL_HANDLE;
	VkBuffer       m_micro_buf  = VK_NULL_HANDLE;
	VkDeviceMemory m_micro_mem  = VK_NULL_HANDLE;
	VkBuffer       m_mem_buf    = VK_NULL_HANDLE;
	VkDeviceMemory m_mem_mem    = VK_NULL_HANDLE;

	VkDescriptorSetLayout m_dsl         = VK_NULL_HANDLE;
	VkPipelineLayout      m_pl_layout   = VK_NULL_HANDLE;
	VkPipeline            m_pipeline    = VK_NULL_HANDLE;
	VkDescriptorPool      m_desc_pool   = VK_NULL_HANDLE;
	VkDescriptorSet       m_desc_set    = VK_NULL_HANDLE;
	VkShaderModule        m_shader      = VK_NULL_HANDLE;

	VkDeviceSize m_state_size = 0;
	VkDeviceSize m_micro_size = 0;
	VkDeviceSize m_mem_size   = 0;
#else
	static constexpr bool m_ready = false;
#endif
};

#ifdef ENABLE_VULKAN

static bool SelectComputeQueueFamily(VkPhysicalDevice phys, uint32_t& out_family)
{
	uint32_t count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, nullptr);
	std::vector<VkQueueFamilyProperties> props(count);
	vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, props.data());

	for (uint32_t i = 0; i < count; ++i)
	{
		if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
		{
			out_family = i;
			return true;
		}
	}
	return false;
}

bool GpuVUContext::AllocateMemory(VkBuffer buf, VkDeviceMemory& mem, VkMemoryPropertyFlags props)
{
	VkMemoryRequirements reqs;
	vkGetBufferMemoryRequirements(m_device, buf, &reqs);

	VkPhysicalDeviceMemoryProperties mem_props;
	vkGetPhysicalDeviceMemoryProperties(m_phys_device, &mem_props);

	for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i)
	{
		if ((reqs.memoryTypeBits & (1u << i)) &&
		    (mem_props.memoryTypes[i].propertyFlags & props) == props)
		{
			VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
			ai.allocationSize  = reqs.size;
			ai.memoryTypeIndex = i;
			return vkAllocateMemory(m_device, &ai, nullptr, &mem) == VK_SUCCESS;
		}
	}
	return false;
}

bool GpuVUContext::CreateBuffers(VkDeviceSize stateSize, VkDeviceSize microSize, VkDeviceSize memSize)
{
	const VkMemoryPropertyFlags host_coherent =
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	auto make_buffer = [&](VkDeviceSize size, VkBuffer& buf, VkDeviceMemory& dev_mem) -> bool
	{
		VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		ci.size  = size;
		ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		if (vkCreateBuffer(m_device, &ci, nullptr, &buf) != VK_SUCCESS) return false;
		if (!AllocateMemory(buf, dev_mem, host_coherent))                return false;
		return vkBindBufferMemory(m_device, buf, dev_mem, 0) == VK_SUCCESS;
	};

	m_state_size = stateSize;
	m_micro_size = microSize;
	m_mem_size   = memSize;

	return make_buffer(stateSize, m_state_buf, m_state_mem) &&
	       make_buffer(microSize, m_micro_buf, m_micro_mem) &&
	       make_buffer(memSize,   m_mem_buf,   m_mem_mem);
}

// Embedded GLSL compute shader source (see GpuVU/vu_execute.glsl).
// This string is used when integrating full glslang compilation.
// TODO(gpu-vu): load and compile vu_execute.glsl via glslang.

bool GpuVUContext::CreatePipeline(const std::string& /* unused – shader is embedded */)
{
	// --- Descriptor set layout (3 SSBOs: state, micro, mem) ---
	VkDescriptorSetLayoutBinding bindings[3] = {};
	for (uint32_t i = 0; i < 3; ++i)
	{
		bindings[i].binding         = i;
		bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	VkDescriptorSetLayoutCreateInfo dsl_ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	dsl_ci.bindingCount = 3;
	dsl_ci.pBindings    = bindings;
	if (vkCreateDescriptorSetLayout(m_device, &dsl_ci, nullptr, &m_dsl) != VK_SUCCESS)
		return false;

	// --- Push constants: cycles + vuIdx ---
	VkPushConstantRange pcr;
	pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pcr.offset     = 0;
	pcr.size       = sizeof(uint32_t) * 2;

	VkPipelineLayoutCreateInfo pl_ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	pl_ci.setLayoutCount         = 1;
	pl_ci.pSetLayouts            = &m_dsl;
	pl_ci.pushConstantRangeCount = 1;
	pl_ci.pPushConstantRanges    = &pcr;
	if (vkCreatePipelineLayout(m_device, &pl_ci, nullptr, &m_pl_layout) != VK_SUCCESS)
		return false;

	// --- Descriptor pool + set ---
	VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
	VkDescriptorPoolCreateInfo pool_ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
	pool_ci.maxSets       = 1;
	pool_ci.poolSizeCount = 1;
	pool_ci.pPoolSizes    = &pool_size;
	if (vkCreateDescriptorPool(m_device, &pool_ci, nullptr, &m_desc_pool) != VK_SUCCESS)
		return false;

	VkDescriptorSetAllocateInfo ds_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
	ds_ai.descriptorPool     = m_desc_pool;
	ds_ai.descriptorSetCount = 1;
	ds_ai.pSetLayouts        = &m_dsl;
	if (vkAllocateDescriptorSets(m_device, &ds_ai, &m_desc_set) != VK_SUCCESS)
		return false;

	// Bind the three storage buffers to the descriptor set.
	VkDescriptorBufferInfo buf_infos[3] = {
		{m_state_buf, 0, m_state_size},
		{m_micro_buf, 0, m_micro_size},
		{m_mem_buf,   0, m_mem_size  },
	};
	VkWriteDescriptorSet writes[3] = {};
	for (uint32_t i = 0; i < 3; ++i)
	{
		writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet          = m_desc_set;
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[i].pBufferInfo     = &buf_infos[i];
	}
	vkUpdateDescriptorSets(m_device, 3, writes, 0, nullptr);

	// --- Compile the GLSL compute shader to SPIR-V ---
	// NOTE: In a real shipping build the SPIR-V would be pre-compiled and embedded
	// as a byte array.  Here we rely on PCSX2's existing glslang integration
	// (the same path used by GSDeviceVK) to compile from source at startup.
	// The VKShaderCache class manages shader compilation and caching.
	//
	// Because VKShaderCache is tied to a specific VkDevice managed by the GS thread,
	// we instead compile inline here using a minimal glslang invocation.  If glslang
	// is not available we fall back to the CPU interpreter.
	//
	// The GLSL compute shader source is in GpuVU/vu_execute.glsl.
	// TODO(gpu-vu): integrate full glslang compilation.
	// For now, mark the pipeline as not ready so that the fallback path is taken.
	// This preserves all of the surrounding Vulkan infrastructure while keeping the
	// implementation honest about what is complete.
	Console.Warning("GpuVU: SPIR-V compilation via glslang not yet integrated; "
	                "GPU VU backend will fall back to software interpreter for execution. "
	                "Vulkan context and pipeline infrastructure is initialised.");
	return false;
}

bool GpuVUContext::Initialize(bool isVU1)
{
	m_isVU1 = isVU1;

	if (!Vulkan::IsVulkanLibraryLoaded())
	{
		// Try loading Vulkan independently.
		Error err;
		if (!Vulkan::LoadVulkanLibrary(&err))
		{
			Console.Warning("GpuVU: Vulkan library not available (%s); GPU VU backend disabled.",
			                err.GetDescription().c_str());
			return false;
		}
	}

	// --- Create VkInstance ---
	VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app_info.pApplicationName   = "PCSX2 GpuVU";
	app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	app_info.pEngineName        = "PCSX2";
	app_info.engineVersion      = VK_MAKE_VERSION(2, 0, 0);
	app_info.apiVersion         = VK_API_VERSION_1_1;

	VkInstanceCreateInfo inst_ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	inst_ci.pApplicationInfo = &app_info;

	if (vkCreateInstance(&inst_ci, nullptr, &m_instance) != VK_SUCCESS)
	{
		Console.Warning("GpuVU: vkCreateInstance failed; GPU VU backend disabled.");
		return false;
	}

	if (!Vulkan::LoadVulkanInstanceFunctions(m_instance))
	{
		Console.Warning("GpuVU: Failed to load Vulkan instance functions.");
		Destroy();
		return false;
	}

	// --- Pick first physical device with a compute queue ---
	uint32_t phys_count = 0;
	vkEnumeratePhysicalDevices(m_instance, &phys_count, nullptr);
	if (phys_count == 0)
	{
		Console.Warning("GpuVU: No Vulkan physical devices found.");
		Destroy();
		return false;
	}
	std::vector<VkPhysicalDevice> phys_devs(phys_count);
	vkEnumeratePhysicalDevices(m_instance, &phys_count, phys_devs.data());

	for (auto& pd : phys_devs)
	{
		if (SelectComputeQueueFamily(pd, m_queue_family))
		{
			m_phys_device = pd;
			break;
		}
	}
	if (m_phys_device == VK_NULL_HANDLE)
	{
		Console.Warning("GpuVU: No Vulkan device with a compute queue found.");
		Destroy();
		return false;
	}

	// --- Create VkDevice ---
	const float queue_prio = 1.0f;
	VkDeviceQueueCreateInfo q_ci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	q_ci.queueFamilyIndex = m_queue_family;
	q_ci.queueCount       = 1;
	q_ci.pQueuePriorities = &queue_prio;

	VkDeviceCreateInfo dev_ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	dev_ci.queueCreateInfoCount = 1;
	dev_ci.pQueueCreateInfos    = &q_ci;

	if (vkCreateDevice(m_phys_device, &dev_ci, nullptr, &m_device) != VK_SUCCESS)
	{
		Console.Warning("GpuVU: vkCreateDevice failed.");
		Destroy();
		return false;
	}

	if (!Vulkan::LoadVulkanDeviceFunctions(m_device))
	{
		Console.Warning("GpuVU: Failed to load Vulkan device functions.");
		Destroy();
		return false;
	}

	vkGetDeviceQueue(m_device, m_queue_family, 0, &m_compute_queue);

	// --- Command pool + buffer ---
	VkCommandPoolCreateInfo cp_ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	cp_ci.queueFamilyIndex = m_queue_family;
	cp_ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	if (vkCreateCommandPool(m_device, &cp_ci, nullptr, &m_cmd_pool) != VK_SUCCESS)
	{
		Destroy(); return false;
	}

	VkCommandBufferAllocateInfo cb_ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	cb_ai.commandPool        = m_cmd_pool;
	cb_ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cb_ai.commandBufferCount = 1;
	if (vkAllocateCommandBuffers(m_device, &cb_ai, &m_cmd_buf) != VK_SUCCESS)
	{
		Destroy(); return false;
	}

	// --- Fence ---
	VkFenceCreateInfo fence_ci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	if (vkCreateFence(m_device, &fence_ci, nullptr, &m_fence) != VK_SUCCESS)
	{
		Destroy(); return false;
	}

	// --- Allocate buffers ---
	const VkDeviceSize state_sz = sizeof(GpuVUState);
	const VkDeviceSize micro_sz = isVU1 ? VU1_PROGSIZE : VU0_PROGSIZE;
	const VkDeviceSize mem_sz   = isVU1 ? VU1_MEMSIZE  : VU0_MEMSIZE;
	if (!CreateBuffers(state_sz, micro_sz, mem_sz))
	{
		Console.Warning("GpuVU: Buffer allocation failed.");
		Destroy(); return false;
	}

	// --- Build compute pipeline ---
	if (!CreatePipeline({}))
	{
		// Pipeline creation failed (glslang integration pending).
		// The context still exists but m_ready stays false, so Execute() will
		// transparently use the CPU fallback path.
		Console.WriteLn("GpuVU: Compute pipeline not ready; using CPU fallback.");
		return false;
	}

	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(m_phys_device, &props);
	Console.WriteLn("GpuVU: Compute context initialised on '%s'.", props.deviceName);
	m_ready = true;
	return true;
}

void GpuVUContext::Destroy()
{
	if (m_device != VK_NULL_HANDLE)
	{
		vkDeviceWaitIdle(m_device);
		if (m_shader    != VK_NULL_HANDLE) { vkDestroyShaderModule(m_device, m_shader,    nullptr); m_shader    = VK_NULL_HANDLE; }
		if (m_pipeline  != VK_NULL_HANDLE) { vkDestroyPipeline(m_device, m_pipeline,      nullptr); m_pipeline  = VK_NULL_HANDLE; }
		if (m_pl_layout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_device, m_pl_layout, nullptr); m_pl_layout = VK_NULL_HANDLE; }
		if (m_dsl       != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_device, m_dsl,  nullptr); m_dsl       = VK_NULL_HANDLE; }
		if (m_desc_pool != VK_NULL_HANDLE) { vkDestroyDescriptorPool(m_device, m_desc_pool, nullptr); m_desc_pool = VK_NULL_HANDLE; }
		if (m_state_buf != VK_NULL_HANDLE) { vkDestroyBuffer(m_device, m_state_buf, nullptr); m_state_buf = VK_NULL_HANDLE; }
		if (m_micro_buf != VK_NULL_HANDLE) { vkDestroyBuffer(m_device, m_micro_buf, nullptr); m_micro_buf = VK_NULL_HANDLE; }
		if (m_mem_buf   != VK_NULL_HANDLE) { vkDestroyBuffer(m_device, m_mem_buf,   nullptr); m_mem_buf   = VK_NULL_HANDLE; }
		if (m_state_mem != VK_NULL_HANDLE) { vkFreeMemory(m_device, m_state_mem, nullptr); m_state_mem = VK_NULL_HANDLE; }
		if (m_micro_mem != VK_NULL_HANDLE) { vkFreeMemory(m_device, m_micro_mem, nullptr); m_micro_mem = VK_NULL_HANDLE; }
		if (m_mem_mem   != VK_NULL_HANDLE) { vkFreeMemory(m_device, m_mem_mem,   nullptr); m_mem_mem   = VK_NULL_HANDLE; }
		if (m_fence     != VK_NULL_HANDLE) { vkDestroyFence(m_device, m_fence, nullptr);   m_fence     = VK_NULL_HANDLE; }
		if (m_cmd_pool  != VK_NULL_HANDLE) { vkDestroyCommandPool(m_device, m_cmd_pool, nullptr); m_cmd_pool = VK_NULL_HANDLE; }
		vkDestroyDevice(m_device, nullptr);
		m_device = VK_NULL_HANDLE;
	}
	if (m_instance != VK_NULL_HANDLE)
	{
		vkDestroyInstance(m_instance, nullptr);
		m_instance = VK_NULL_HANDLE;
	}
	m_ready = false;
}

void GpuVUContext::CopyVURegsToState(const VURegs* vu, bool /*isVU1*/, GpuVUState& s) const
{
	for (int i = 0; i < 32; ++i)
	{
		s.VF[i][0] = vu->VF[i].f.x;
		s.VF[i][1] = vu->VF[i].f.y;
		s.VF[i][2] = vu->VF[i].f.z;
		s.VF[i][3] = vu->VF[i].f.w;
		s.VI[i]    = vu->VI[i].UL & 0xFFFFu;
	}
	s.ACC[0]   = vu->ACC.f.x;
	s.ACC[1]   = vu->ACC.f.y;
	s.ACC[2]   = vu->ACC.f.z;
	s.ACC[3]   = vu->ACC.f.w;
	s.Q        = vu->q.F;
	s.P        = vu->p.F;
	s.TPC      = vu->VI[REG_TPC].UL;
	s.cycle    = static_cast<uint32_t>(vu->cycle);
	s.ebit     = vu->ebit;
	s.macflag  = vu->macflag;
	s.statusflag = vu->statusflag;
	s.clipflag = vu->clipflag;
	s.flags    = vu->flags;
	s.needs_cpu_fallback = 0;
	s._pad[0] = s._pad[1] = 0;
}

void GpuVUContext::CopyStateToVURegs(const GpuVUState& s, VURegs* vu, bool /*isVU1*/) const
{
	for (int i = 0; i < 32; ++i)
	{
		vu->VF[i].f.x = s.VF[i][0];
		vu->VF[i].f.y = s.VF[i][1];
		vu->VF[i].f.z = s.VF[i][2];
		vu->VF[i].f.w = s.VF[i][3];
		vu->VI[i].UL  = s.VI[i] & 0xFFFFu;
	}
	vu->ACC.f.x  = s.ACC[0];
	vu->ACC.f.y  = s.ACC[1];
	vu->ACC.f.z  = s.ACC[2];
	vu->ACC.f.w  = s.ACC[3];
	vu->q.F      = s.Q;
	vu->p.F      = s.P;
	vu->VI[REG_TPC].UL = s.TPC;
	vu->cycle    = s.cycle;
	vu->ebit     = s.ebit;
	vu->macflag  = s.macflag;
	vu->statusflag = s.statusflag;
	vu->clipflag = s.clipflag;
	vu->flags    = s.flags;
}

void GpuVUContext::Reset(VURegs* vu, bool isVU1)
{
	if (!m_device) return;

	// Zero the GPU-side state buffer.
	GpuVUState empty = {};
	CopyVURegsToState(vu, isVU1, empty);

	void* ptr = nullptr;
	if (vkMapMemory(m_device, m_state_mem, 0, sizeof(GpuVUState), 0, &ptr) == VK_SUCCESS)
	{
		std::memcpy(ptr, &empty, sizeof(GpuVUState));
		vkUnmapMemory(m_device, m_state_mem);
	}
}

bool GpuVUContext::Execute(VURegs* vu, uint32_t cycles, bool isVU1)
{
	if (!m_ready || m_pipeline == VK_NULL_HANDLE)
		return false; // caller will use the CPU fallback

	void* state_ptr = nullptr;
	void* micro_ptr = nullptr;
	void* mem_ptr   = nullptr;

	if (vkMapMemory(m_device, m_state_mem, 0, sizeof(GpuVUState),  0, &state_ptr) != VK_SUCCESS ||
	    vkMapMemory(m_device, m_micro_mem, 0, m_micro_size,         0, &micro_ptr) != VK_SUCCESS ||
	    vkMapMemory(m_device, m_mem_mem,   0, m_mem_size,           0, &mem_ptr)   != VK_SUCCESS)
	{
		return false;
	}

	// Upload VU state, microcode, and data memory.
	GpuVUState gpu_state;
	CopyVURegsToState(vu, isVU1, gpu_state);
	std::memcpy(state_ptr, &gpu_state, sizeof(GpuVUState));
	std::memcpy(micro_ptr, vu->Micro, static_cast<size_t>(m_micro_size));
	std::memcpy(mem_ptr,   vu->Mem,   static_cast<size_t>(m_mem_size));

	vkUnmapMemory(m_device, m_state_mem);
	vkUnmapMemory(m_device, m_micro_mem);
	vkUnmapMemory(m_device, m_mem_mem);

	// Record and submit the compute command.
	VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(m_cmd_buf, &cbi);

	vkCmdBindPipeline(m_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
	vkCmdBindDescriptorSets(m_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
	                        m_pl_layout, 0, 1, &m_desc_set, 0, nullptr);

	uint32_t push[2] = {cycles, isVU1 ? 1u : 0u};
	vkCmdPushConstants(m_cmd_buf, m_pl_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
	vkCmdDispatch(m_cmd_buf, 1, 1, 1);

	vkEndCommandBuffer(m_cmd_buf);

	VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
	si.commandBufferCount = 1;
	si.pCommandBuffers    = &m_cmd_buf;
	if (vkQueueSubmit(m_compute_queue, 1, &si, m_fence) != VK_SUCCESS)
		return false;

	vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
	vkResetFences(m_device, 1, &m_fence);

	// Read back results.
	if (vkMapMemory(m_device, m_state_mem, 0, sizeof(GpuVUState), 0, &state_ptr) != VK_SUCCESS)
		return false;

	std::memcpy(&gpu_state, state_ptr, sizeof(GpuVUState));
	vkUnmapMemory(m_device, m_state_mem);

	if (gpu_state.needs_cpu_fallback)
	{
		// The shader encountered an unimplemented instruction.  Restore the original
		// VU state (the caller will re-run the program on the CPU interpreter).
		return false;
	}

	// Write updated state back.
	if (vkMapMemory(m_device, m_mem_mem, 0, m_mem_size, 0, &mem_ptr) == VK_SUCCESS)
	{
		std::memcpy(vu->Mem, mem_ptr, static_cast<size_t>(m_mem_size));
		vkUnmapMemory(m_device, m_mem_mem);
	}

	CopyStateToVURegs(gpu_state, vu, isVU1);
	return true;
}

#else // !ENABLE_VULKAN

bool GpuVUContext::Initialize(bool)  { return false; }
void GpuVUContext::Destroy()         {}
bool GpuVUContext::Execute(VURegs*, uint32_t, bool) { return false; }
void GpuVUContext::Reset(VURegs*, bool)             {}

#endif // ENABLE_VULKAN

// ---------------------------------------------------------------------------
//  GpuVUmicro0
// ---------------------------------------------------------------------------

GpuVUmicro0::GpuVUmicro0()
{
	m_Idx         = 0;
	IsInterpreter = false;
}

GpuVUmicro0::~GpuVUmicro0()
{
	Shutdown();
}

bool GpuVUmicro0::EnsureContext()
{
	if (m_ctx)
		return m_ctx->IsReady();

	m_ctx = std::make_shared<GpuVUContext>();
	m_ctx->Initialize(/*isVU1=*/false);
	return m_ctx->IsReady();
}

void GpuVUmicro0::Shutdown()
{
	m_ctx.reset();
	m_fallback.Shutdown();
}

void GpuVUmicro0::Reset()
{
	m_fallback.Reset();
	if (m_ctx)
		m_ctx->Reset(&VU0, /*isVU1=*/false);
}

void GpuVUmicro0::SetStartPC(u32 startPC)
{
	m_fallback.SetStartPC(startPC);
}

void GpuVUmicro0::Execute(u32 cycles)
{
	if (EnsureContext() && m_ctx->Execute(&VU0, cycles, /*isVU1=*/false))
		return;

	// GPU path unavailable or encountered unimplemented instructions – use CPU.
	m_fallback.Execute(cycles);
}

void GpuVUmicro0::Step()
{
	m_fallback.Step();
}

void GpuVUmicro0::Clear(u32 addr, u32 size)
{
	m_fallback.Clear(addr, size);
}

// ---------------------------------------------------------------------------
//  GpuVUmicro1
// ---------------------------------------------------------------------------

GpuVUmicro1::GpuVUmicro1()
{
	m_Idx         = 1;
	IsInterpreter = false;
}

GpuVUmicro1::~GpuVUmicro1()
{
	Shutdown();
}

bool GpuVUmicro1::EnsureContext()
{
	if (m_ctx)
		return m_ctx->IsReady();

	m_ctx = std::make_shared<GpuVUContext>();
	m_ctx->Initialize(/*isVU1=*/true);
	return m_ctx->IsReady();
}

void GpuVUmicro1::Shutdown()
{
	m_ctx.reset();
	m_fallback.Shutdown();
}

void GpuVUmicro1::Reset()
{
	m_fallback.Reset();
	if (m_ctx)
		m_ctx->Reset(&VU1, /*isVU1=*/true);
}

void GpuVUmicro1::SetStartPC(u32 startPC)
{
	m_fallback.SetStartPC(startPC);
}

void GpuVUmicro1::Execute(u32 cycles)
{
	if (EnsureContext() && m_ctx->Execute(&VU1, cycles, /*isVU1=*/true))
		return;

	m_fallback.Execute(cycles);
}

void GpuVUmicro1::Step()
{
	m_fallback.Step();
}

void GpuVUmicro1::Clear(u32 addr, u32 size)
{
	m_fallback.Clear(addr, size);
}

void GpuVUmicro1::ResumeXGkick()
{
	m_fallback.ResumeXGkick();
}

// ---------------------------------------------------------------------------
//  Plugin registration
// ---------------------------------------------------------------------------

void RegisterGpuVUBackend()
{
	VUPluginRegistry::Register({
		VUBackendType::GPU,
		"gpuVU",
		"GPU Compute (Vulkan)",
#ifdef ENABLE_VULKAN
		[]() -> bool { return true; },
#else
		[]() -> bool { return false; },
#endif
		[]() -> std::unique_ptr<BaseVUmicroCPU> { return std::make_unique<GpuVUmicro0>(); },
		[]() -> std::unique_ptr<BaseVUmicroCPU> { return std::make_unique<GpuVUmicro1>(); },
	});
}

// Called from VUPluginRegistry::RegisterBuiltins() (defined in VUmicro.cpp).
namespace VUPluginRegistry
{
	void RegisterGpuBackend()
	{
		RegisterGpuVUBackend();
	}
}
