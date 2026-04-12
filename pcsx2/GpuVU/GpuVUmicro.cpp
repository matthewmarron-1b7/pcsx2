// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

// --------------------------------------------------------------------------------------
//  GpuVUmicro - Vulkan compute-shader VU backend (implementation)
// --------------------------------------------------------------------------------------
//
// This file provides:
//   1. GpuVUContext  - a minimal, self-contained Vulkan compute context that is wholly
//      independent of the GS rendering thread.  It owns its own VkInstance, VkDevice,
//      compute VkQueue, command pool, and all pipeline / buffer resources.
//   2. GpuVUmicro0 / GpuVUmicro1  - BaseVUmicroCPU implementations that delegate VU
//      execution to the context via a single vkCmdDispatch call.
//   3. RegisterGpuVUBackend()  - registers the GPU descriptor with VUPluginRegistry.
//
// Fallback strategy
// -----------------
// When Vulkan initialisation fails (Vulkan runtime absent, no suitable device, ...) the
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
#include "GS/Renderers/Vulkan/VKShaderCache.h"
#include <cstring>
#include <optional>
#include <vector>
#include <string>
#endif

// ---------------------------------------------------------------------------
//  GpuVUState  (must match the layout in vu_execute.glsl)
// ---------------------------------------------------------------------------

struct alignas(16) GpuVUState
{
	float    VF[32][4];      // 32 x vec4
	uint32_t VI[32];         // 32 x uint (only low 16 bits are valid)
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

// Embedded GLSL compute shader source for the VU compute pipeline.
// Kept in sync with GpuVU/vu_execute.glsl.
// clang-format off
static const char VU_EXECUTE_GLSL[] = R"GLSL(
// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+
//
// vu_execute.glsl - Vulkan compute shader: PS2 Vector Unit microcode interpreter
//
// This shader is dispatched with a single work-group of one thread.  It reads the
// complete VU state from a storage buffer, simulates up to 'push.cycles' VU clock
// cycles worth of instruction pairs (each pair = one upper-slot + one lower-slot
// instruction), and writes the updated state back.
//
// The shader implements the complete PS2 VU instruction set.  Instructions that
// require host-side interaction (XGKICK, XTOP, XITOP) set the
// 'needs_cpu_fallback' flag so that the C++ fallback path takes over.
//
// --- Instruction encoding (both words are 32-bit) ---
//   Upper word (micro[pc/4 + 1]):
//     bit 31     I-bit  - lower word is a 32-bit float immediate → I register
//     bit 30     E-bit  - last instruction of the micro-program
//     bit 29     M-bit  - (VU0) sets VUFLAG_MFLAGSET
//     bit 28     D-bit  - debug break
//     bit 27     T-bit  - debug break
//     bits [25:21] dest field (XYZW mask)
//     bits [20:16] Ft register
//     bits [15:11] Fs register
//     bits [10:6]  Fd register
//     bits  [5:0]  upper opcode
//
//   Lower word (micro[pc/4]):
//     bits [31:25] primary lower opcode
//     bits [24:21] dest / special field
//     bits [20:16] Ft / It register
//     bits [15:11] Fs / Is register
//     bits [10:6]  Fd / Id register
//     bits  [5:0]  secondary lower opcode (when primary == 0x40)

#version 450
// Re-enable explicit 32-bit type helpers as required by earlier shader revision.
// All integer arithmetic uses standard 'int'/'uint' which already provide 32-bit
// semantics in GLSL 4.50, so the extension is kept for forward compatibility.
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// VU register state (must match GpuVUState in GpuVUContext.h)
struct GpuVUState
{
vec4  VF[32];          // VF[0] hardwired to (0,0,0,1) by convention
uint  VI[32];          // VI[0] hardwired to 0.  Full 32 bits stored for
                       //   VI[20]=R, VI[21]=I, VI[22]=Q-bits, VI[23]=P-bits.
vec4  ACC;
float Q;               // Q register (DIV/SQRT/RSQRT result, as float)
float P;               // P register (EFU result, as float)
uint  TPC;             // Thread Program Counter (byte address)
uint  cycle;
uint  ebit;            // E-bit countdown (2→1→0=stop)
uint  macflag;
uint  statusflag;
uint  clipflag;
uint  flags;
uint  needs_cpu_fallback;
uint  _pad[2];
};

layout(std430, binding = 0) buffer VUStateBuffer { GpuVUState vu; };
layout(std430, binding = 1) buffer VUMicroBuffer { uint micro[]; };
layout(std430, binding = 2) buffer VUMemBuffer   { uint mem[];   };

layout(push_constant) uniform PushConstants
{
uint cycles;
uint vuIdx;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const float PS2_MAX_FLOAT = uintBitsToFloat(0x7F7FFFFFu);

float ps2_clamp(float v)
{
if (isnan(v)) return 0.0;
if (isinf(v)) return (v > 0.0) ? PS2_MAX_FLOAT : -PS2_MAX_FLOAT;
// Flush denormals to zero
uint bits = floatBitsToUint(v);
if ((bits & 0x7F800000u) == 0u) return 0.0;
return v;
}

vec4 ps2_clamp4(vec4 v)
{
return vec4(ps2_clamp(v.x), ps2_clamp(v.y), ps2_clamp(v.z), ps2_clamp(v.w));
}

// Field extractors
uint dest_field(uint op) { return (op >> 21) & 0xFu; }
uint ft_field(uint op)   { return (op >> 16) & 0x1Fu; }
uint fs_field(uint op)   { return (op >> 11) & 0x1Fu; }
uint fd_field(uint op)   { return (op >>  6) & 0x1Fu; }
uint upper_op(uint op)   { return op & 0x3Fu; }
uint lower_op(uint op)   { return (op >> 25) & 0x7Fu; }

// Select VF component by 2-bit field selector (0=X, 1=Y, 2=Z, 3=W)
float vf_comp(vec4 v, uint sel)
{
return (sel == 0u) ? v.x : (sel == 1u) ? v.y : (sel == 2u) ? v.z : v.w;
}

// Read a component of VF[r] selected by bits [22:21] of op (fsf field)
float vf_fsf(uint r, uint op) { return vf_comp(vu.VF[r], (op >> 21u) & 0x3u); }
// Read a component of VF[r] selected by bits [24:23] of op (ftf field)
float vf_ftf(uint r, uint op) { return vf_comp(vu.VF[r], (op >> 23u) & 0x3u); }

// Apply XYZW dest mask
vec4 apply_dest(vec4 dst, vec4 src, uint mask)
{
return vec4(
(mask & 8u) != 0u ? src.x : dst.x,
(mask & 4u) != 0u ? src.y : dst.y,
(mask & 2u) != 0u ? src.z : dst.z,
(mask & 1u) != 0u ? src.w : dst.w);
}

// Update MAC flags (simplified – update Z/S/O/U per component)
void update_mac_flags(vec4 result, uint dest)
{
uint mf = vu.macflag;
// Shift all flags left by 4 to make room for new flags
// (abbreviated: just update sign bits for now)
if ((dest & 8u) != 0u) {
mf = (mf & ~0xC0u) | (result.x < 0.0 ? 0x80u : 0u) | (result.x == 0.0 ? 0x40u : 0u);
}
if ((dest & 4u) != 0u) {
mf = (mf & ~0x0Cu) | (result.y < 0.0 ? 0x08u : 0u) | (result.y == 0.0 ? 0x04u : 0u);
}
if ((dest & 2u) != 0u) {
mf = (mf & ~0x30u) | (result.z < 0.0 ? 0x20u : 0u) | (result.z == 0.0 ? 0x10u : 0u);
}
if ((dest & 1u) != 0u) {
mf = (mf & ~0x03u) | (result.w < 0.0 ? 0x02u : 0u) | (result.w == 0.0 ? 0x01u : 0u);
}
vu.macflag = mf;
}

// Write result to VF[fd] (never writes to fd==0)
void write_vf(uint fd, vec4 result, uint dest)
{
if (fd != 0u)
vu.VF[fd] = apply_dest(vu.VF[fd], result, dest);
update_mac_flags(result, dest);
}

// Write result to ACC
void write_acc(vec4 result, uint dest)
{
vu.ACC = apply_dest(vu.ACC, result, dest);
update_mac_flags(result, dest);
}

// Read VI register as signed 16-bit value (VI[0] always returns 0)
int vi_s16(uint r) { return r == 0u ? 0 : (int(vu.VI[r] << 16u) >> 16); }

// Write VI register (never writes to r==0).
// Preserves full 32 bits for special registers (VI[20]=R, VI[21]=I,
// VI[22]=Q-bits, VI[23]=P-bits); masks to 16 bits for general integer registers.
void write_vi(uint r, uint val)
{
if (r == 0u) return;
vu.VI[r] = (r >= 20u && r <= 23u) ? val : (val & 0xFFFFu);
}

// Read current I register (full 32-bit float, stored in VI[21])
float i_reg() { return uintBitsToFloat(vu.VI[21]); }

// FTOI helper: clamp float to signed 32-bit integer range and return bits
uint ftoi_clamp(float f)
{
uint bits = floatBitsToUint(f);
if ((bits & 0x7F800000u) >= 0x4F000000u)
return (bits & 0x80000000u) != 0u ? 0x80000000u : 0x7FFFFFFFu;
return uint(int(f));
}

// ---------------------------------------------------------------------------
// Upper-slot FD secondary decode (opcodes 0x3C–0x3F)
// Handles accumulator ops, ABS, CLIP, ITOF/FTOI, NOP
// ---------------------------------------------------------------------------

void exec_upper_fd(uint opcode, uint op)
{
uint fd_idx  = opcode & 0x3u;     // 0=_00(x bc), 1=_01(y bc), 2=_10(z bc), 3=_11(w bc)
uint fd_op   = (op >> 6u) & 0x1Fu; // tertiary index [10:6]
uint dest    = dest_field(op);
uint ft      = ft_field(op);
uint fs      = fs_field(op);
uint fd      = fd_field(op);

vec4 vfs = vu.VF[fs];
vec4 vft = vu.VF[ft];
float I  = i_reg();

// Broadcast: component selected by fd_idx
float bc = (fd_idx == 0u) ? vft.x : (fd_idx == 1u) ? vft.y :
           (fd_idx == 2u) ? vft.z : vft.w;

switch (fd_op)
{
// --- fd_op 0x00: ADDAx/y/z/w  ACC = VF[fs] + VF[ft].{xyzw} ---
case 0x00u: write_acc(ps2_clamp4(vfs + vec4(bc)), dest); return;
// --- fd_op 0x01: SUBAx/y/z/w  ACC = VF[fs] - VF[ft].{xyzw} ---
case 0x01u: write_acc(ps2_clamp4(vfs - vec4(bc)), dest); return;
// --- fd_op 0x02: MADDAx/y/z/w  ACC += VF[fs] * VF[ft].{xyzw} ---
case 0x02u: write_acc(ps2_clamp4(vu.ACC + vfs * vec4(bc)), dest); return;
// --- fd_op 0x03: MSUBAx/y/z/w  ACC -= VF[fs] * VF[ft].{xyzw} ---
case 0x03u: write_acc(ps2_clamp4(vu.ACC - vfs * vec4(bc)), dest); return;
// --- fd_op 0x04: ITOF0/4/12/15 (integer→float with scale) ---
case 0x04u:
{
// fd_idx selects ITOF0(x-bc), ITOF4(y-bc), ITOF12(z-bc), ITOF15(w-bc)
const float scales[4] = float[4](1.0, 1.0/16.0, 1.0/4096.0, 1.0/32768.0);
float scale = scales[fd_idx];
vec4 r = vec4(
float(int(floatBitsToUint(vfs.x))) * scale,
float(int(floatBitsToUint(vfs.y))) * scale,
float(int(floatBitsToUint(vfs.z))) * scale,
float(int(floatBitsToUint(vfs.w))) * scale);
write_vf(ft, r, dest);
return;
}
// --- fd_op 0x05: FTOI0/4/12/15 (float→integer with scale) ---
case 0x05u:
{
const float scales[4] = float[4](1.0, 16.0, 4096.0, 32768.0);
float scale = scales[fd_idx];
vec4 r = vec4(
uintBitsToFloat(ftoi_clamp(vfs.x * scale)),
uintBitsToFloat(ftoi_clamp(vfs.y * scale)),
uintBitsToFloat(ftoi_clamp(vfs.z * scale)),
uintBitsToFloat(ftoi_clamp(vfs.w * scale)));
write_vf(ft, r, dest);
return;
}
// --- fd_op 0x06: MULAx/y/z/w  ACC = VF[fs] * VF[ft].{xyzw} ---
case 0x06u: write_acc(ps2_clamp4(vfs * vec4(bc)), dest); return;
// --- fd_op 0x07: per fd_idx: MULAq/ABS/MULAi/CLIP ---
case 0x07u:
if (fd_idx == 0u) { // MULAq: ACC = VF[fs] * Q
write_acc(ps2_clamp4(vfs * vu.Q), dest);
} else if (fd_idx == 1u) { // ABS: VF[ft] = |VF[fs]|
write_vf(ft, abs(vfs), dest);
} else if (fd_idx == 2u) { // MULAi: ACC = VF[fs] * I
write_acc(ps2_clamp4(vfs * I), dest);
} else { // CLIP: update clip flag
float w = abs(vft.w);
uint cf = vu.clipflag << 6u;
uint bits = floatBitsToUint(w);
// Treat denormal w as smallest denormal for comparisons
if ((bits & 0x7F800000u) == 0u) bits = 0x007FFFFFu;
uint px = floatBitsToUint(vfs.x);
uint nx = floatBitsToUint(-vfs.x); // flip sign for negative test
uint py = floatBitsToUint(vfs.y);
uint ny = floatBitsToUint(-vfs.y);
uint pz = floatBitsToUint(vfs.z);
uint nz = floatBitsToUint(-vfs.z);
if (int(px & 0x7FFFFFFFu) > int(bits)) cf |= 0x01u;
if (int(nx & 0x7FFFFFFFu) > int(bits)) cf |= 0x02u;
if (int(py & 0x7FFFFFFFu) > int(bits)) cf |= 0x04u;
if (int(ny & 0x7FFFFFFFu) > int(bits)) cf |= 0x08u;
if (int(pz & 0x7FFFFFFFu) > int(bits)) cf |= 0x10u;
if (int(nz & 0x7FFFFFFFu) > int(bits)) cf |= 0x20u;
vu.clipflag = cf & 0xFFFFFFu;
}
return;
// --- fd_op 0x08: ADDAq/MADDAq/ADDAi/MADDAi ---
case 0x08u:
if (fd_idx == 0u)      write_acc(ps2_clamp4(vfs + vu.Q), dest);     // ADDAq
else if (fd_idx == 1u) write_acc(ps2_clamp4(vu.ACC + vfs * vu.Q), dest); // MADDAq
else if (fd_idx == 2u) write_acc(ps2_clamp4(vfs + I), dest);         // ADDAi
else                   write_acc(ps2_clamp4(vu.ACC + vfs * I), dest); // MADDAi
return;
// --- fd_op 0x09: SUBAq/MSUBAq/SUBAi/MSUBAi ---
case 0x09u:
if (fd_idx == 0u)      write_acc(ps2_clamp4(vfs - vu.Q), dest);       // SUBAq
else if (fd_idx == 1u) write_acc(ps2_clamp4(vu.ACC - vfs * vu.Q), dest); // MSUBAq
else if (fd_idx == 2u) write_acc(ps2_clamp4(vfs - I), dest);           // SUBAi
else                   write_acc(ps2_clamp4(vu.ACC - vfs * I), dest);  // MSUBAi
return;
// --- fd_op 0x0A: ADDA/MADDA/MULA/OPMULA ---
case 0x0Au:
if (fd_idx == 0u) { // ADDA: ACC = VF[fs] + VF[ft]
write_acc(ps2_clamp4(vfs + vft), dest);
} else if (fd_idx == 1u) { // MADDA: ACC += VF[fs] * VF[ft]
write_acc(ps2_clamp4(vu.ACC + vfs * vft), dest);
} else if (fd_idx == 2u) { // MULA: ACC = VF[fs] * VF[ft]
write_acc(ps2_clamp4(vfs * vft), dest);
} else { // OPMULA: ACC.xyz = cross product components
vec4 r = vu.ACC;
r.x = ps2_clamp(vfs.y * vft.z);
r.y = ps2_clamp(vfs.z * vft.x);
r.z = ps2_clamp(vfs.x * vft.y);
vu.ACC = r;
// OPMULA always writes XYZ, ignoring dest mask
}
return;
// --- fd_op 0x0B: SUBA/MSUBA/unknown/NOP ---
case 0x0Bu:
if (fd_idx == 0u) { // SUBA: ACC = VF[fs] - VF[ft]
write_acc(ps2_clamp4(vfs - vft), dest);
} else if (fd_idx == 1u) { // MSUBA: ACC -= VF[fs] * VF[ft]
write_acc(ps2_clamp4(vu.ACC - vfs * vft), dest);
} else if (fd_idx == 3u) { // NOP
// no-op
} else {
vu.needs_cpu_fallback = 1u;
}
return;
default:
vu.needs_cpu_fallback = 1u;
return;
}
}

// ---------------------------------------------------------------------------
// Upper-slot execution (FMAC, bits [5:0] of upper word)
// ---------------------------------------------------------------------------

void exec_upper(uint op)
{
uint opcode = upper_op(op);
uint dest   = dest_field(op);
uint ft     = ft_field(op);
uint fs     = fs_field(op);
uint fd     = fd_field(op);

vec4 vfs = vu.VF[fs];
vec4 vft = vu.VF[ft];
vec4 result;

switch (opcode)
{
// ADDx/y/z/w
case 0x00u: result = ps2_clamp4(vfs + vec4(vft.x)); break;
case 0x01u: result = ps2_clamp4(vfs + vec4(vft.y)); break;
case 0x02u: result = ps2_clamp4(vfs + vec4(vft.z)); break;
case 0x03u: result = ps2_clamp4(vfs + vec4(vft.w)); break;
// SUBx/y/z/w
case 0x04u: result = ps2_clamp4(vfs - vec4(vft.x)); break;
case 0x05u: result = ps2_clamp4(vfs - vec4(vft.y)); break;
case 0x06u: result = ps2_clamp4(vfs - vec4(vft.z)); break;
case 0x07u: result = ps2_clamp4(vfs - vec4(vft.w)); break;
// MADDx/y/z/w
case 0x08u: result = ps2_clamp4(vu.ACC + vfs * vec4(vft.x)); break;
case 0x09u: result = ps2_clamp4(vu.ACC + vfs * vec4(vft.y)); break;
case 0x0Au: result = ps2_clamp4(vu.ACC + vfs * vec4(vft.z)); break;
case 0x0Bu: result = ps2_clamp4(vu.ACC + vfs * vec4(vft.w)); break;
// MSUBx/y/z/w
case 0x0Cu: result = ps2_clamp4(vu.ACC - vfs * vec4(vft.x)); break;
case 0x0Du: result = ps2_clamp4(vu.ACC - vfs * vec4(vft.y)); break;
case 0x0Eu: result = ps2_clamp4(vu.ACC - vfs * vec4(vft.z)); break;
case 0x0Fu: result = ps2_clamp4(vu.ACC - vfs * vec4(vft.w)); break;
// MAXx/y/z/w
case 0x10u: result = max(vfs, vec4(vft.x)); break;
case 0x11u: result = max(vfs, vec4(vft.y)); break;
case 0x12u: result = max(vfs, vec4(vft.z)); break;
case 0x13u: result = max(vfs, vec4(vft.w)); break;
// MINIx/y/z/w
case 0x14u: result = min(vfs, vec4(vft.x)); break;
case 0x15u: result = min(vfs, vec4(vft.y)); break;
case 0x16u: result = min(vfs, vec4(vft.z)); break;
case 0x17u: result = min(vfs, vec4(vft.w)); break;
// MULx/y/z/w
case 0x18u: result = ps2_clamp4(vfs * vec4(vft.x)); break;
case 0x19u: result = ps2_clamp4(vfs * vec4(vft.y)); break;
case 0x1Au: result = ps2_clamp4(vfs * vec4(vft.z)); break;
case 0x1Bu: result = ps2_clamp4(vfs * vec4(vft.w)); break;
// MULq / MAXi / MULi / MINIi
case 0x1Cu: result = ps2_clamp4(vfs * vu.Q);          break; // MULq
case 0x1Du: result = max(vfs, vec4(i_reg()));          break; // MAXi
case 0x1Eu: result = ps2_clamp4(vfs * i_reg());        break; // MULi
case 0x1Fu: result = min(vfs, vec4(i_reg()));          break; // MINIi
// ADDq / MADDq / ADDi / MADDi
case 0x20u: result = ps2_clamp4(vfs + vu.Q);           break; // ADDq
case 0x21u: result = ps2_clamp4(vu.ACC + vfs * vu.Q);  break; // MADDq
case 0x22u: result = ps2_clamp4(vfs + i_reg());         break; // ADDi
case 0x23u: result = ps2_clamp4(vu.ACC + vfs * i_reg()); break; // MADDi
// SUBq / MSUBq / SUBi / MSUBi
case 0x24u: result = ps2_clamp4(vfs - vu.Q);           break; // SUBq
case 0x25u: result = ps2_clamp4(vu.ACC - vfs * vu.Q);  break; // MSUBq
case 0x26u: result = ps2_clamp4(vfs - i_reg());         break; // SUBi
case 0x27u: result = ps2_clamp4(vu.ACC - vfs * i_reg()); break; // MSUBi
// ADD / MADD / MUL / MAX
case 0x28u: result = ps2_clamp4(vfs + vft);            break; // ADD
case 0x29u: result = ps2_clamp4(vu.ACC + vfs * vft);   break; // MADD
case 0x2Au: result = ps2_clamp4(vfs * vft);            break; // MUL
case 0x2Bu: result = max(vfs, vft);                    break; // MAX
// SUB / MSUB / OPMSUB / MINI
case 0x2Cu: result = ps2_clamp4(vfs - vft); break; // SUB
case 0x2Du: result = ps2_clamp4(vu.ACC - vfs * vft); break; // MSUB
case 0x2Eu: // OPMSUB: VF[fd].xyz = ACC.xyz - cross(VF[fs].xyz, VF[ft].xyz) (component form)
{
// OPMSUB: fd.x=ACC.x-fs.y*ft.z, fd.y=ACC.y-fs.z*ft.x, fd.z=ACC.z-fs.x*ft.y
vec4 r = vu.VF[fd];
r.x = ps2_clamp(vu.ACC.x - vfs.y * vft.z);
r.y = ps2_clamp(vu.ACC.y - vfs.z * vft.x);
r.z = ps2_clamp(vu.ACC.z - vfs.x * vft.y);
if (fd != 0u) vu.VF[fd] = r;
update_mac_flags(r, 0xEu); // XYZ always
return;
}
case 0x2Fu: result = min(vfs, vft); break; // MINI
// FD secondary decode
case 0x3Cu: exec_upper_fd(opcode, op); return;
case 0x3Du: exec_upper_fd(opcode, op); return;
case 0x3Eu: exec_upper_fd(opcode, op); return;
case 0x3Fu: exec_upper_fd(opcode, op); return;
default:
vu.needs_cpu_fallback = 1u;
return;
}

write_vf(fd, result, dest);
}

// ---------------------------------------------------------------------------
// Lower-slot T3 (tertiary) decode – called from exec_lower_op
// t3_idx: 0=T3_00(x-bc), 1=T3_01(y-bc), 2=T3_10(z-bc), 3=T3_11(w-bc)
// tertiary: (op >> 6) & 0x1F
// ---------------------------------------------------------------------------

void exec_lower_t3(uint t3_idx, uint tertiary, uint op)
{
uint dest = dest_field(op);
uint ft   = ft_field(op);
uint fs   = fs_field(op);

switch (tertiary)
{
// --------------- index 0x0C ---------------
case 0x0Cu:
if (t3_idx == 0u) {
// MOVE: VF[ft].dest = VF[fs]
if (ft != 0u)
vu.VF[ft] = apply_dest(vu.VF[ft], vu.VF[fs], dest);
} else if (t3_idx == 1u) {
// MR32: VF[ft].dest = rotate-left(VF[fs]): x←y, y←z, z←w, w←x
if (ft != 0u) {
vec4 r = vec4(vu.VF[fs].y, vu.VF[fs].z, vu.VF[fs].w, vu.VF[fs].x);
vu.VF[ft] = apply_dest(vu.VF[ft], r, dest);
}
} else {
vu.needs_cpu_fallback = 1u;
}
return;

// --------------- index 0x0D ---------------
case 0x0Du:
{
// T3_00: LQI  VF[ft] = mem[VI[fs]]; VI[fs]++
// T3_01: SQI  mem[VI[ft]] = VF[fs]; VI[ft]++
// T3_10: LQD  VI[fs]--; VF[ft] = mem[VI[fs]]
// T3_11: SQD  VI[ft]--; mem[VI[ft]] = VF[fs]
uint addr_reg;
bool is_store = (t3_idx == 1u) || (t3_idx == 3u);
bool pre_dec  = (t3_idx == 2u) || (t3_idx == 3u);
bool post_inc = (t3_idx == 0u) || (t3_idx == 1u);

if (is_store) {
addr_reg = ft;
} else {
addr_reg = fs;
}

if (pre_dec && addr_reg != 0u) {
vu.VI[addr_reg] = (vu.VI[addr_reg] - 1u) & 0xFFFFu;
}

uint base_addr = (vu.VI[addr_reg] & 0xFFFFu) * 16u / 4u;

if (!is_store) {
// LQI / LQD: load from mem
if (base_addr + 3u < uint(mem.length())) {
vec4 val = vec4(
uintBitsToFloat(mem[base_addr + 0u]),
uintBitsToFloat(mem[base_addr + 1u]),
uintBitsToFloat(mem[base_addr + 2u]),
uintBitsToFloat(mem[base_addr + 3u]));
if (ft != 0u)
vu.VF[ft] = apply_dest(vu.VF[ft], val, dest);
}
} else {
// SQI / SQD: store to mem
if (base_addr + 3u < uint(mem.length())) {
if ((dest & 8u) != 0u) mem[base_addr + 0u] = floatBitsToUint(vu.VF[fs].x);
if ((dest & 4u) != 0u) mem[base_addr + 1u] = floatBitsToUint(vu.VF[fs].y);
if ((dest & 2u) != 0u) mem[base_addr + 2u] = floatBitsToUint(vu.VF[fs].z);
if ((dest & 1u) != 0u) mem[base_addr + 3u] = floatBitsToUint(vu.VF[fs].w);
}
}

if (post_inc && addr_reg != 0u) {
vu.VI[addr_reg] = (vu.VI[addr_reg] + 1u) & 0xFFFFu;
}
return;
}

// --------------- index 0x0E ---------------
case 0x0Eu:
if (t3_idx == 0u) {
// DIV: Q = VF[fs].fsf / VF[ft].ftf
float num = vf_fsf(fs, op);
float den = vf_ftf(ft, op);
if (den == 0.0)
vu.Q = (num >= 0.0) ? PS2_MAX_FLOAT : -PS2_MAX_FLOAT;
else
vu.Q = ps2_clamp(num / den);
} else if (t3_idx == 1u) {
// SQRT: Q = sqrt(|VF[ft].ftf|)
float v = abs(vf_ftf(ft, op));
vu.Q = sqrt(v);
} else if (t3_idx == 2u) {
// RSQRT: Q = VF[fs].fsf / sqrt(|VF[ft].ftf|)
float num = vf_fsf(fs, op);
float den = abs(vf_ftf(ft, op));
if (den == 0.0)
vu.Q = (num >= 0.0) ? PS2_MAX_FLOAT : -PS2_MAX_FLOAT;
else
vu.Q = ps2_clamp(num / sqrt(den));
} else {
// WAITQ: stall until Q pipe done - NOP in GPU (Q is always ready)
}
return;

// --------------- index 0x0F ---------------
case 0x0Fu:
if (t3_idx == 0u) {
// MTIR: VI[ft].lower = VF[fs].fsf (truncate float to 16-bit int)
float src = vf_fsf(fs, op);
write_vi(ft, uint(int(src)) & 0xFFFFu);
} else if (t3_idx == 1u) {
// MFIR: VF[ft].dest = float(sign_extend_16(VI[fs]))
int sval = vi_s16(fs);
if (ft != 0u)
vu.VF[ft] = apply_dest(vu.VF[ft], vec4(float(sval)), dest);
} else if (t3_idx == 2u) {
// ILWR: VI[ft] = mem_int[VI[fs]].field
uint base = (vu.VI[fs] & 0xFFFFu) * 16u / 4u;
uint field = (op >> 21u) & 0x3u; // dest bits [1:0] select field
// Actually field is encoded in dest bits: X=bit3, Y=bit2, Z=bit1, W=bit0
// The field to read: use the highest set bit in dest
uint sel = 0u;
if ((dest & 8u) != 0u) sel = 0u;
else if ((dest & 4u) != 0u) sel = 1u;
else if ((dest & 2u) != 0u) sel = 2u;
else sel = 3u;
if (base + sel < uint(mem.length()))
write_vi(ft, mem[base + sel] & 0xFFFFu);
} else {
// ISWR: mem_int[VI[ft]].field = VI[fs]
uint base = (vu.VI[ft] & 0xFFFFu) * 16u / 4u;
for (uint b = 0u; b < 4u; ++b) {
if ((dest & (8u >> b)) != 0u && base + b < uint(mem.length()))
mem[base + b] = vu.VI[fs] & 0xFFFFu;
}
}
return;

// --------------- index 0x10 ---------------
case 0x10u:
{
// T3_00: RNEXT  VF[ft].dest = R; AdvanceLFSR
// T3_01: RGET   VF[ft].dest = R
// T3_10: RINIT  R = 0x3F800000 | (VF[fs].fsf & 0x7FFFFF)
// T3_11: RXOR   R = 0x3F800000 | ((R ^ VF[fs].fsf) & 0x7FFFFF)
if (t3_idx == 0u || t3_idx == 1u) {
if (t3_idx == 0u) {
// Advance LFSR before reading
uint r = vu.VI[20];
uint x = (r >> 4u) & 1u;
uint y = (r >> 22u) & 1u;
r <<= 1u;
r ^= x ^ y;
r = (r & 0x7FFFFFu) | 0x3F800000u;
vu.VI[20] = r;
}
// RGET (and RNEXT after advance): copy R to VF[ft].dest
float r_val = uintBitsToFloat(vu.VI[20]);
if (ft != 0u)
vu.VF[ft] = apply_dest(vu.VF[ft], vec4(r_val), dest);
} else if (t3_idx == 2u) {
// RINIT
uint src_bits = floatBitsToUint(vf_fsf(fs, op));
vu.VI[20] = 0x3F800000u | (src_bits & 0x7FFFFFu);
} else {
// RXOR
uint src_bits = floatBitsToUint(vf_fsf(fs, op));
vu.VI[20] = 0x3F800000u | ((vu.VI[20] ^ src_bits) & 0x7FFFFFu);
}
return;
}

// --------------- index 0x19 ---------------
case 0x19u:
if (t3_idx == 0u) {
// MFP: VF[ft].dest = P (VU1 only)
if (ft != 0u)
vu.VF[ft] = apply_dest(vu.VF[ft], vec4(vu.P), dest);
} else {
vu.needs_cpu_fallback = 1u;
}
return;

// --------------- index 0x1A ---------------
case 0x1Au:
// T3_00: XTOP / T3_01: XITOP / T3_10: unknown
// These interact with VIF/GIF - require CPU fallback
vu.needs_cpu_fallback = 1u;
return;

// --------------- index 0x1B ---------------
case 0x1Bu:
if (t3_idx == 0u) {
// XGKICK: kick GIF transfer - requires host interaction
vu.needs_cpu_fallback = 1u;
} else {
vu.needs_cpu_fallback = 1u;
}
return;

// --------------- index 0x1C ---------------
case 0x1Cu:
{
float x = vu.VF[fs].x, y = vu.VF[fs].y, z = vu.VF[fs].z;
if (t3_idx == 0u) {
// ESADD: P = x^2 + y^2 + z^2
vu.P = ps2_clamp(x*x + y*y + z*z);
} else if (t3_idx == 1u) {
// ERSADD: P = 1 / (x^2 + y^2 + z^2)
float s = x*x + y*y + z*z;
vu.P = (s == 0.0) ? PS2_MAX_FLOAT : ps2_clamp(1.0 / s);
} else if (t3_idx == 2u) {
// ELENG: P = sqrt(x^2 + y^2 + z^2)
vu.P = ps2_clamp(sqrt(x*x + y*y + z*z));
} else {
// ERLENG: P = 1 / sqrt(x^2 + y^2 + z^2)
float l = sqrt(x*x + y*y + z*z);
vu.P = (l == 0.0) ? PS2_MAX_FLOAT : ps2_clamp(1.0 / l);
}
return;
}

// --------------- index 0x1D ---------------
case 0x1Du:
if (t3_idx == 0u) {
// EATANxy: P = atan(VF[fs].y / VF[fs].x)
vu.P = ps2_clamp(atan(vu.VF[fs].y, vu.VF[fs].x));
} else if (t3_idx == 1u) {
// EATANxz: P = atan(VF[fs].z / VF[fs].x)
vu.P = ps2_clamp(atan(vu.VF[fs].z, vu.VF[fs].x));
} else if (t3_idx == 2u) {
// ESUM: P = x + y + z + w
vec4 v = vu.VF[fs];
vu.P = ps2_clamp(v.x + v.y + v.z + v.w);
} else {
vu.needs_cpu_fallback = 1u;
}
return;

// --------------- index 0x1E ---------------
case 0x1Eu:
if (t3_idx == 0u) {
// ESQRT: P = sqrt(|VF[fs].fsf|)
vu.P = sqrt(abs(vf_fsf(fs, op)));
} else if (t3_idx == 1u) {
// ERSQRT: P = 1 / sqrt(|VF[fs].fsf|)
float v = abs(vf_fsf(fs, op));
vu.P = (v == 0.0) ? PS2_MAX_FLOAT : ps2_clamp(1.0 / sqrt(v));
} else if (t3_idx == 2u) {
// ERCPR: P = 1 / VF[fs].fsf
float v = vf_fsf(fs, op);
vu.P = (v == 0.0) ? PS2_MAX_FLOAT : ps2_clamp(1.0 / v);
} else {
// WAITP: stall until P pipe done - NOP in GPU
}
return;

// --------------- index 0x1F ---------------
case 0x1Fu:
if (t3_idx == 0u) {
// ESIN: P = sin(VF[fs].fsf)  (PS2 uses a Padé approximation; GLSL sin is close enough)
vu.P = ps2_clamp(sin(vf_fsf(fs, op)));
} else if (t3_idx == 1u) {
// EATAN: P = atan(VF[fs].fsf)
vu.P = ps2_clamp(atan(vf_fsf(fs, op)));
} else if (t3_idx == 2u) {
// EEXP: PS2 VU polynomial approximation  P = 1/((1+sum(consts[i]*x^i))^4)
float eexp_x = vf_fsf(fs, op);
float ep = 1.0
    + 0.249998688697815 * eexp_x
    + 0.031257584691048 * eexp_x * eexp_x
    + 0.002591371303424 * eexp_x * eexp_x * eexp_x
    + 0.000171562001924 * eexp_x * eexp_x * eexp_x * eexp_x
    + 0.000005430199963 * eexp_x * eexp_x * eexp_x * eexp_x * eexp_x
    + 0.000000690600018 * eexp_x * eexp_x * eexp_x * eexp_x * eexp_x * eexp_x;
ep = ep * ep; ep = ep * ep; // pow(ep, 4)
vu.P = (ep == 0.0) ? PS2_MAX_FLOAT : ps2_clamp(1.0 / ep);
} else {
vu.needs_cpu_fallback = 1u;
}
return;

default:
vu.needs_cpu_fallback = 1u;
return;
}
}

// ---------------------------------------------------------------------------
// Lower-slot secondary decode – primary opcode 0x40 → LowerOP table
// secondary: op & 0x3F (bits [5:0])
// ---------------------------------------------------------------------------

void exec_lower_op(uint op)
{
uint secondary = op & 0x3Fu;
uint dest = dest_field(op);
uint ft   = ft_field(op);
uint fs   = fs_field(op);
uint fd   = fd_field(op);

switch (secondary)
{
// IADD: VI[fd] = VI[fs] + VI[ft]  (signed 16-bit)
case 0x30u:
write_vi(fd, uint(vi_s16(fs) + vi_s16(ft)));
return;
// ISUB: VI[fd] = VI[fs] - VI[ft]
case 0x31u:
write_vi(fd, uint(vi_s16(fs) - vi_s16(ft)));
return;
// IADDI: VI[ft] = VI[fs] + imm5 (sign-extended 5-bit)
case 0x32u:
{
int imm5 = int((op >> 6u) & 0x1Fu);
if ((imm5 & 0x10) != 0) imm5 |= int(0xFFFFFFE0u);
write_vi(ft, uint(vi_s16(fs) + imm5));
return;
}
// IAND: VI[fd] = VI[fs] & VI[ft]
case 0x34u:
write_vi(fd, vu.VI[fs] & vu.VI[ft]);
return;
// IOR: VI[fd] = VI[fs] | VI[ft]
case 0x35u:
write_vi(fd, vu.VI[fs] | vu.VI[ft]);
return;
// T3 tertiary dispatch
case 0x3Cu: exec_lower_t3(0u, (op >> 6u) & 0x1Fu, op); return; // T3_00
case 0x3Du: exec_lower_t3(1u, (op >> 6u) & 0x1Fu, op); return; // T3_01
case 0x3Eu: exec_lower_t3(2u, (op >> 6u) & 0x1Fu, op); return; // T3_10
case 0x3Fu: exec_lower_t3(3u, (op >> 6u) & 0x1Fu, op); return; // T3_11
default:
vu.needs_cpu_fallback = 1u;
return;
}
}

// ---------------------------------------------------------------------------
// Lower-slot primary decode
// ---------------------------------------------------------------------------

void exec_lower(uint op)
{
uint opcode = lower_op(op);  // bits [31:25]
uint dest   = dest_field(op);
uint ft     = ft_field(op);
uint fs     = fs_field(op);

// Sign-extend 11-bit immediate (bits [10:0]) – used by branches
int imm11 = int(op & 0x7FFu);
if ((op & 0x400u) != 0u) imm11 |= int(0xFFFFF800u);

switch (opcode)
{
// ---- LQ: VF[ft].dest = mem128[VI[fs] + imm11] ----
case 0x00u:
{
uint addr = uint(int(vu.VI[fs] & 0xFFFFu) + imm11) & 0xFFFFu;
addr = addr * 16u / 4u;
if (addr + 3u < uint(mem.length())) {
vec4 val = vec4(uintBitsToFloat(mem[addr]),
                uintBitsToFloat(mem[addr+1u]),
                uintBitsToFloat(mem[addr+2u]),
                uintBitsToFloat(mem[addr+3u]));
if (ft != 0u)
vu.VF[ft] = apply_dest(vu.VF[ft], val, dest);
}
return;
}
// ---- SQ: mem128[VI[ft] + imm11] = VF[fs] ----
case 0x01u:
{
uint addr = uint(int(vu.VI[ft] & 0xFFFFu) + imm11) & 0xFFFFu;
addr = addr * 16u / 4u;
if (addr + 3u < uint(mem.length())) {
if ((dest & 8u) != 0u) mem[addr]    = floatBitsToUint(vu.VF[fs].x);
if ((dest & 4u) != 0u) mem[addr+1u] = floatBitsToUint(vu.VF[fs].y);
if ((dest & 2u) != 0u) mem[addr+2u] = floatBitsToUint(vu.VF[fs].z);
if ((dest & 1u) != 0u) mem[addr+3u] = floatBitsToUint(vu.VF[fs].w);
}
return;
}
// ---- ILW: VI[ft].elem = mem_int16[VI[fs] + imm11].elem ----
case 0x04u:
{
uint addr = uint(int(vu.VI[fs] & 0xFFFFu) + imm11) & 0xFFFFu;
addr = addr * 16u / 4u;
// dest field encodes which element: X=3(highest), Y=2, Z=1, W=0(lowest)
uint sel = 0u;
if      ((dest & 8u) != 0u) sel = 0u;
else if ((dest & 4u) != 0u) sel = 1u;
else if ((dest & 2u) != 0u) sel = 2u;
else                        sel = 3u;
if (addr + sel < uint(mem.length()))
write_vi(ft, mem[addr + sel] & 0xFFFFu);
return;
}
// ---- ISW: mem_int16[VI[ft] + imm11].elem = VI[fs] ----
case 0x05u:
{
uint addr = uint(int(vu.VI[ft] & 0xFFFFu) + imm11) & 0xFFFFu;
addr = addr * 16u / 4u;
for (uint b = 0u; b < 4u; ++b) {
if ((dest & (8u >> b)) != 0u && addr + b < uint(mem.length()))
mem[addr + b] = vu.VI[fs] & 0xFFFFu;
}
return;
}
// ---- IADDIU: VI[ft] = VI[fs] + imm15 ----
case 0x08u:
{
uint imm_bits = ((op >> 10u) & 0x7800u) | (op & 0x7FFu);
int imm15 = int(imm_bits);
if ((imm15 & 0x4000) != 0) imm15 |= int(0xFFFF8000u);
write_vi(ft, uint(vi_s16(fs) + imm15));
return;
}
// ---- ISUBIU: VI[ft] = VI[fs] - imm15 ----
case 0x09u:
{
uint imm_bits = ((op >> 10u) & 0x7800u) | (op & 0x7FFu);
int imm15 = int(imm_bits);
if ((imm15 & 0x4000) != 0) imm15 |= int(0xFFFF8000u);
write_vi(ft, uint(vi_s16(fs) - imm15));
return;
}
// ---- Flag condition checks (FCEQ/FCSET/FCAND/FCOR/FSEQ/FSSET/FSAND/FSOR/FMEQ/FMAND/FMOR/FCGET) ----
case 0x10u: // FCEQ: VI[1] = (clipflag & 0xFFFFFF == (imm24 & 0xFFFFFF)) ? 1 : 0
write_vi(1u, ((vu.clipflag & 0xFFFFFFu) == (op & 0xFFFFFFu)) ? 1u : 0u);
return;
case 0x11u: // FCSET: clipflag = imm24
vu.clipflag = op & 0xFFFFFFu;
return;
case 0x12u: // FCAND: VI[1] = ((clipflag & 0xFFFFFF) & (imm24)) != 0 ? 1 : 0
write_vi(1u, ((vu.clipflag & op & 0xFFFFFFu) != 0u) ? 1u : 0u);
return;
case 0x13u: // FCOR: VI[1] = ((clipflag | imm24) & 0xFFFFFF) == 0xFFFFFF ? 1 : 0
write_vi(1u, ((vu.clipflag | op) & 0xFFFFFFu) == 0xFFFFFFu ? 1u : 0u);
return;
case 0x14u: // FSEQ: VI[ft] = (statusflag & 0xFF) == (imm8 & 0xFF) ? 1 : 0
{
uint imm = (op & 0x800u) != 0u ? ((op & 0xF00u) | (op & 0xFFu)) : (op & 0xFFu);
write_vi(ft, (vu.statusflag & 0xFFu) == (imm & 0xFFu) ? 1u : 0u);
return;
}
case 0x15u: // FSSET: statusflag = (statusflag & 0x3F) | (imm12 & 0xFC0)
vu.statusflag = (op & 0xFC0u) | (vu.statusflag & 0x03Fu);
return;
case 0x16u: // FSAND: VI[ft] = statusflag & (imm12 & 0xFFF)
write_vi(ft, (vu.statusflag & (op & 0xFFFu)) & 0xFFFFu);
return;
case 0x17u: // FSOR: VI[ft] = statusflag | (imm12 & 0xFFF)
write_vi(ft, (vu.statusflag | (op & 0xFFFu)) & 0xFFFFu);
return;
case 0x18u: // FMEQ: VI[ft] = (macflag & 0xFFFF) == (imm16 & 0xFFFF) ? 1 : 0
write_vi(ft, (vu.macflag & 0xFFFFu) == (op & 0xFFFFu) ? 1u : 0u);
return;
case 0x1Au: // FMAND: VI[ft] = macflag & (imm12 & 0xFFF)
write_vi(ft, (vu.macflag & (op & 0xFFFu)) & 0xFFFFu);
return;
case 0x1Bu: // FMOR: VI[ft] = macflag | (imm12 & 0xFFF)
write_vi(ft, (vu.macflag | (op & 0xFFFu)) & 0xFFFFu);
return;
case 0x1Cu: // FCGET: VI[ft] = clipflag & 0xFFF
write_vi(ft, vu.clipflag & 0xFFFu);
return;
// ---- B: TPC += imm11 * 8 (unconditional branch) ----
case 0x20u:
vu.TPC = uint(int(vu.TPC) + imm11 * 8);
return;
// ---- BAL: VI[ft] = (TPC + 8) / 8; TPC += imm11 * 8 ----
case 0x21u:
write_vi(ft, (vu.TPC + 8u) / 8u);
vu.TPC = uint(int(vu.TPC) + imm11 * 8);
return;
// ---- JR: TPC = VI[fs] * 8 ----
case 0x24u:
vu.TPC = (vu.VI[fs] & 0xFFFFu) * 8u;
return;
// ---- JALR: VI[ft] = (TPC + 8) / 8; TPC = VI[fs] * 8 ----
case 0x25u:
write_vi(ft, (vu.TPC + 8u) / 8u);
vu.TPC = (vu.VI[fs] & 0xFFFFu) * 8u;
return;
// ---- IBEQ: if VI[fs] == VI[ft]: TPC += imm11 * 8 ----
case 0x28u:
if ((vu.VI[fs] & 0xFFFFu) == (vu.VI[ft] & 0xFFFFu))
vu.TPC = uint(int(vu.TPC) + imm11 * 8);
return;
// ---- IBNE: if VI[fs] != VI[ft]: TPC += imm11 * 8 ----
case 0x29u:
if ((vu.VI[fs] & 0xFFFFu) != (vu.VI[ft] & 0xFFFFu))
vu.TPC = uint(int(vu.TPC) + imm11 * 8);
return;
// ---- IBLTZ: if VI[fs] < 0: TPC += imm11 * 8 ----
case 0x2Cu:
if (vi_s16(fs) < 0)
vu.TPC = uint(int(vu.TPC) + imm11 * 8);
return;
// ---- IBGTZ: if VI[fs] > 0: TPC += imm11 * 8 ----
case 0x2Du:
if (vi_s16(fs) > 0)
vu.TPC = uint(int(vu.TPC) + imm11 * 8);
return;
// ---- IBLEZ: if VI[fs] <= 0: TPC += imm11 * 8 ----
case 0x2Eu:
if (vi_s16(fs) <= 0)
vu.TPC = uint(int(vu.TPC) + imm11 * 8);
return;
// ---- IBGEZ: if VI[fs] >= 0: TPC += imm11 * 8 ----
case 0x2Fu:
if (vi_s16(fs) >= 0)
vu.TPC = uint(int(vu.TPC) + imm11 * 8);
return;
// ---- LowerOP secondary decode ----
case 0x40u:
exec_lower_op(op);
return;
default:
vu.needs_cpu_fallback = 1u;
return;
}
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

void main()
{
vu.needs_cpu_fallback = 0u;

uint micro_size = uint(micro.length()) * 4u;
uint run_bit    = (vuIdx == 0u) ? 0x1u : 0x100u;

for (uint i = 0u; i < cycles; ++i)
{
// Check if the VU is still running
if ((vu.VI[29] & run_bit) == 0u)
break;

// E-bit countdown
if (vu.ebit > 0u)
{
vu.ebit -= 1u;
if (vu.ebit == 0u)
{
vu.VI[29] &= ~run_bit;
break;
}
}

uint pc = vu.TPC;
if (pc + 7u >= micro_size)
{
vu.VI[29] &= ~run_bit;
break;
}

uint word_idx   = pc / 4u;
uint lower_word = micro[word_idx];
uint upper_word = micro[word_idx + 1u];

// Advance TPC before executing (branch instructions override this)
vu.TPC = pc + 8u;

// Check E-bit (end of micro-program)
if ((upper_word & 0x40000000u) != 0u)
vu.ebit = 2u;

// Execute upper slot (FMAC)
exec_upper(upper_word);
if (vu.needs_cpu_fallback != 0u)
return;

// Check I-flag (bit 31 of upper word): lower word is a 32-bit float immediate
if ((upper_word & 0x80000000u) != 0u)
{
// Store lower word as the I register (full 32-bit float bits)
vu.VI[21] = lower_word;
}
else
{
exec_lower(lower_word);
if (vu.needs_cpu_fallback != 0u)
return;
}

vu.cycle += 1u;
}
}

)GLSL";
// clang-format on

bool GpuVUContext::CreatePipeline(const std::string& /* unused - shader is embedded */)
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

	// --- Compile the embedded GLSL compute shader to SPIR-V ---
	// VKShaderCache::CompileComputeShaderToSPV uses shaderc internally.
	// If shaderc is unavailable (e.g., stripped binary) the optional is empty,
	// and we fall back to the CPU interpreter.
	auto spirv = VKShaderCache::CompileComputeShaderToSPV(VU_EXECUTE_GLSL, false);
	if (!spirv.has_value() || spirv->empty())
	{
		Console.Warning("GpuVU: SPIR-V compilation failed; GPU VU backend will use CPU fallback.");
		return false;
	}

	// --- Create VkShaderModule from SPIR-V ---
	VkShaderModuleCreateInfo sm_ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	sm_ci.codeSize = spirv->size() * sizeof(uint32_t);
	sm_ci.pCode    = spirv->data();
	if (vkCreateShaderModule(m_device, &sm_ci, nullptr, &m_shader) != VK_SUCCESS)
	{
		Console.Warning("GpuVU: vkCreateShaderModule failed.");
		return false;
	}

	// --- Create VkComputePipeline ---
	VkPipelineShaderStageCreateInfo stage_ci{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
	stage_ci.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	stage_ci.module = m_shader;
	stage_ci.pName  = "main";

	VkComputePipelineCreateInfo cp_ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
	cp_ci.stage  = stage_ci;
	cp_ci.layout = m_pl_layout;
	if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &m_pipeline) != VK_SUCCESS)
	{
		Console.Warning("GpuVU: vkCreateComputePipelines failed.");
		return false;
	}

	return true;
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
		Console.Warning("GpuVU: Compute pipeline creation failed; using CPU fallback.");
		Destroy();
		return false;
	}

	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(m_phys_device, &props);
	Console.WriteLn("GpuVU: Compute context initialized on '%s'.", props.deviceName);
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
		// General integer registers (VI[0..19]) are 16-bit; special registers
		// (VI[20]=R, VI[21]=I, VI[22]=Q-int, VI[23]=P-int, VI[24..31]) need full
		// 32 bits to preserve float bit patterns in I/R registers.
		s.VI[i] = (i < 20 || i > 23) ? (vu->VI[i].UL & 0xFFFFu) : vu->VI[i].UL;
	}
	// VI[21] (REG_I) stores the 32-bit float bits of the I register. Override
	// it with the actual q/p floats and the full I value to ensure correctness.
	s.VI[21] = vu->VI[REG_I].UL;   // full 32-bit float bits for I register
	s.VI[20] = vu->VI[REG_R].UL;   // full 32-bit LFSR value for R register
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
		// Restore full 32 bits for special registers; mask to 16 for general ones.
		vu->VI[i].UL = (i < 20 || i > 23) ? (s.VI[i] & 0xFFFFu) : s.VI[i];
	}
	// Restore I and R registers from the full 32-bit values
	vu->VI[REG_I].UL = s.VI[21];
	vu->VI[REG_R].UL = s.VI[20];
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

	// GPU path unavailable or encountered unimplemented instructions - use CPU.
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
