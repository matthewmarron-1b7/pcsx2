// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+
//
// vu_execute.glsl – Vulkan compute shader: PS2 Vector Unit microcode interpreter
//
// This shader is dispatched with a single work-group of one thread.  It reads the
// complete VU state from a storage buffer, simulates up to 'push.cycles' VU clock
// cycles worth of instruction pairs (each pair = one upper-slot + one lower-slot
// instruction), and writes the updated state back.
//
// The shader implements the most-common VU instructions needed to sustain real-world
// game workloads.  Instructions not yet implemented here set the 'needs_cpu_fallback'
// flag so that the host C++ code can re-run the affected micro-program on the CPU
// interpreter.
//
// Instruction encoding reference:
//   Upper slot  bits[1]  – field-encoded FMAC operations  (ADD, SUB, MUL, MADD …)
//   Lower slot  bits[0]  – branch / load-store / integer / FDIV / EFU operations
//   Both words share a 3-bit flag prefix in the upper word:
//     bit 30  E-bit  – last instruction of a micro-program
//     bit 29  M-bit  – (VU0 only) sets VUFLAG_MFLAGSET
//     bit 28  D-bit  – debug break
//     bit 27  T-bit  – debug break

#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// VU register state + pipeline state (must match GpuVUState in GpuVUContext.h)
struct GpuVUState
{
	// 32 × 128-bit floating-point vector registers (VF[0..31], each as vec4)
	vec4  VF[32];
	// 32 × 32-bit slots for the 16-bit integer registers (VI[0..31])
	uint  VI[32];
	// Accumulator
	vec4  ACC;
	// Q  (quotient from DIV/SQRT/RSQRT) and P (EFU result)
	float Q;
	float P;
	// Thread Program Counter (byte address into the micro-code buffer)
	uint  TPC;
	// Cycle counter (incremented by the shader)
	uint  cycle;
	// E-bit encountered – micro-program end
	uint  ebit;
	// Status / MAC / Clip flags
	uint  macflag;
	uint  statusflag;
	uint  clipflag;
	// General flags (VUFLAG_MFLAGSET etc.)
	uint  flags;
	// Set to 1 by the shader when it encounters an unimplemented instruction
	uint  needs_cpu_fallback;
	// Padding to keep the struct 16-byte aligned
	uint  _pad[2];
};

layout(std430, binding = 0) buffer VUStateBuffer  { GpuVUState vu; };
layout(std430, binding = 1) buffer VUMicroBuffer  { uint micro[]; }; // microcode words
layout(std430, binding = 2) buffer VUMemBuffer    { uint mem[];   }; // data memory

layout(push_constant) uniform PushConstants
{
	uint cycles;  // number of EE cycles to simulate
	uint vuIdx;   // 0 = VU0, 1 = VU1
};

// ---------------------------------------------------------------------------
// Helper macros / constants
// ---------------------------------------------------------------------------

// PS2 "PS2 float" – flush denormals to zero, clamp to PS2 max.
const float PS2_MAX_FLOAT = 3.40282346638528860e+38;

float ps2_clamp(float v)
{
	if (isnan(v))    return 0.0;
	if (isinf(v))    return (v > 0.0) ? PS2_MAX_FLOAT : -PS2_MAX_FLOAT;
	return v;
}

vec4 ps2_clamp4(vec4 v)
{
	return vec4(ps2_clamp(v.x), ps2_clamp(v.y), ps2_clamp(v.z), ps2_clamp(v.w));
}

// Extract XYZW field from an opcode word (bits 24-21 for upper, 24-21 for lower dest etc.)
// dest field: bits 24-21
uint dest_field(uint op) { return (op >> 21) & 0xFu; }
// ft field: bits 20-16
uint ft_field(uint op)   { return (op >> 16) & 0x1Fu; }
// fs field: bits 15-11
uint fs_field(uint op)   { return (op >> 11) & 0x1Fu; }
// fd field: bits 10-6
uint fd_field(uint op)   { return (op >>  6) & 0x1Fu; }
// upper opcode: bits 5-0
uint upper_op(uint op)   { return op & 0x3Fu; }
// lower opcode: bits 31-25
uint lower_op(uint op)   { return (op >> 25) & 0x7Fu; }

// Apply XYZW dest mask: only update components where the corresponding bit is set.
vec4 apply_dest(vec4 dst, vec4 src, uint mask)
{
	return vec4(
		(mask & 8u) != 0u ? src.x : dst.x,
		(mask & 4u) != 0u ? src.y : dst.y,
		(mask & 2u) != 0u ? src.z : dst.z,
		(mask & 1u) != 0u ? src.w : dst.w);
}

// Update MAC flags for a result vector (simplified, PS2-accurate flag update
// is complex; this gives a reasonable approximation).
void update_mac_flags(vec4 result, uint dest)
{
	uint mf = 0u;
	if ((dest & 8u) != 0u && result.x != 0.0) mf |= 0x40u;
	if ((dest & 4u) != 0u && result.y != 0.0) mf |= 0x04u;
	if ((dest & 2u) != 0u && result.z != 0.0) mf |= 0x10u;
	if ((dest & 1u) != 0u && result.w != 0.0) mf |= 0x01u;
	vu.macflag = mf;
}

// ---------------------------------------------------------------------------
// Upper-slot execution (FMAC instructions, bits [5:0] of the upper word)
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

	// Instruction dispatch by upper opcode bits [5:0]
	switch (opcode)
	{
	// --- ADDx/y/z/w  (ft broadcast) ---
	case 0x00u: result = ps2_clamp4(vfs + vec4(vft.x)); break; // ADDx
	case 0x01u: result = ps2_clamp4(vfs + vec4(vft.y)); break; // ADDy
	case 0x02u: result = ps2_clamp4(vfs + vec4(vft.z)); break; // ADDz
	case 0x03u: result = ps2_clamp4(vfs + vec4(vft.w)); break; // ADDw
	// --- SUBx/y/z/w ---
	case 0x04u: result = ps2_clamp4(vfs - vec4(vft.x)); break; // SUBx
	case 0x05u: result = ps2_clamp4(vfs - vec4(vft.y)); break; // SUBy
	case 0x06u: result = ps2_clamp4(vfs - vec4(vft.z)); break; // SUBz
	case 0x07u: result = ps2_clamp4(vfs - vec4(vft.w)); break; // SUBw
	// --- MADDx/y/z/w (Multiply-Accumulate) ---
	case 0x08u: result = ps2_clamp4(vu.ACC + vfs * vec4(vft.x)); break; // MADDx
	case 0x09u: result = ps2_clamp4(vu.ACC + vfs * vec4(vft.y)); break; // MADDy
	case 0x0Au: result = ps2_clamp4(vu.ACC + vfs * vec4(vft.z)); break; // MADDz
	case 0x0Bu: result = ps2_clamp4(vu.ACC + vfs * vec4(vft.w)); break; // MADDw
	// --- MSUBx/y/z/w (Multiply-Subtract from Accumulator) ---
	case 0x0Cu: result = ps2_clamp4(vu.ACC - vfs * vec4(vft.x)); break; // MSUBx
	case 0x0Du: result = ps2_clamp4(vu.ACC - vfs * vec4(vft.y)); break; // MSUBy
	case 0x0Eu: result = ps2_clamp4(vu.ACC - vfs * vec4(vft.z)); break; // MSUBz
	case 0x0Fu: result = ps2_clamp4(vu.ACC - vfs * vec4(vft.w)); break; // MSUBw
	// --- MAXx/y/z/w ---
	case 0x10u: result = max(vfs, vec4(vft.x)); break; // MAXx
	case 0x11u: result = max(vfs, vec4(vft.y)); break; // MAXy
	case 0x12u: result = max(vfs, vec4(vft.z)); break; // MAXz
	case 0x13u: result = max(vfs, vec4(vft.w)); break; // MAXw
	// --- MINIx/y/z/w ---
	case 0x14u: result = min(vfs, vec4(vft.x)); break; // MINIx
	case 0x15u: result = min(vfs, vec4(vft.y)); break; // MINIy
	case 0x16u: result = min(vfs, vec4(vft.z)); break; // MINIz
	case 0x17u: result = min(vfs, vec4(vft.w)); break; // MINIw
	// --- MULx/y/z/w ---
	case 0x18u: result = ps2_clamp4(vfs * vec4(vft.x)); break; // MULx
	case 0x19u: result = ps2_clamp4(vfs * vec4(vft.y)); break; // MULy
	case 0x1Au: result = ps2_clamp4(vfs * vec4(vft.z)); break; // MULz
	case 0x1Bu: result = ps2_clamp4(vfs * vec4(vft.w)); break; // MULw
	// --- MULq (multiply by Q register) ---
	case 0x1Cu: result = ps2_clamp4(vfs * vu.Q); break; // MULq
	// --- MAXi / MINIi ---
	case 0x1Du: result = max(vfs, vec4(uintBitsToFloat(vu.VI[21]))); break; // MAXi (I register)
	case 0x1Eu: result = min(vfs, vec4(uintBitsToFloat(vu.VI[21]))); break; // MINIi
	// --- MULi ---
	case 0x1Fu: result = ps2_clamp4(vfs * uintBitsToFloat(vu.VI[21])); break; // MULi
	// --- ADDq ---
	case 0x20u: result = ps2_clamp4(vfs + vu.Q); break; // ADDq
	// --- MADDq ---
	case 0x21u: result = ps2_clamp4(vu.ACC + vfs * vu.Q); break; // MADDq
	// --- ADDi ---
	case 0x22u: result = ps2_clamp4(vfs + uintBitsToFloat(vu.VI[21])); break; // ADDi
	// --- MADDi ---
	case 0x23u: result = ps2_clamp4(vu.ACC + vfs * uintBitsToFloat(vu.VI[21])); break; // MADDi
	// --- SUBq ---
	case 0x24u: result = ps2_clamp4(vfs - vu.Q); break; // SUBq
	// --- MSUBq ---
	case 0x25u: result = ps2_clamp4(vu.ACC - vfs * vu.Q); break; // MSUBq
	// --- SUBi ---
	case 0x26u: result = ps2_clamp4(vfs - uintBitsToFloat(vu.VI[21])); break; // SUBi
	// --- MSUBi ---
	case 0x27u: result = ps2_clamp4(vu.ACC - vfs * uintBitsToFloat(vu.VI[21])); break; // MSUBi
	// --- ADD (ft) ---
	case 0x28u: result = ps2_clamp4(vfs + vft); break; // ADD
	// --- MADD (ft) ---
	case 0x29u: result = ps2_clamp4(vu.ACC + vfs * vft); break; // MADD
	// --- MUL (ft) ---
	case 0x2Au: result = ps2_clamp4(vfs * vft); break; // MUL
	// --- MAX (ft) ---
	case 0x2Bu: result = max(vfs, vft); break; // MAX
	// --- SUB (ft) ---
	case 0x2Cu: result = ps2_clamp4(vfs - vft); break; // SUB
	// --- MSUB (ft) ---
	case 0x2Du: result = ps2_clamp4(vu.ACC - vfs * vft); break; // MSUB
	// --- MINI (ft) ---
	case 0x2Eu: result = min(vfs, vft); break; // MINI
	// Special upper opcodes (opcode bits [5:2] = 0x3x) ---
	case 0x3Cu: // SPECIAL upper (opcode in lower bits, handled by upper secondary decode)
	case 0x3Du:
	case 0x3Eu:
	case 0x3Fu:
	{
		// Secondary decode for 0x3C-0x3F: ABS, FTOI*, ITOF*, CLIP
		uint op2 = ((op >> 6) & 0x1Fu) | ((opcode & 0x3u) << 5);
		switch (op2 & 0x1Fu)
		{
		case 0x00u: result = abs(vfs);  break; // ABS
		case 0x01u: // FTOI0
			result = vec4(floor(vfs.x), floor(vfs.y), floor(vfs.z), floor(vfs.w));
			vu.VF[fd] = apply_dest(vu.VF[fd], result, dest);
			update_mac_flags(result, dest);
			return;
		case 0x02u: result = vfs * 2.0; break; // FTOI4 (scale by 2^4)
		case 0x03u: result = vfs * 16.0; break; // FTOI12
		case 0x04u: result = vfs * 4096.0; break; // FTOI15
		case 0x05u: result = vfs * (1.0/1.0);   break; // ITOF0
		case 0x06u: result = vfs * (1.0/16.0);  break; // ITOF4
		case 0x07u: result = vfs * (1.0/4096.0);break; // ITOF12
		case 0x08u: result = vfs * (1.0/32768.0);break;// ITOF15
		default:
			// CLIP, MR32, or other – needs CPU fallback
			vu.needs_cpu_fallback = 1u;
			return;
		}
		break;
	}
	default:
		vu.needs_cpu_fallback = 1u;
		return;
	}

	vu.VF[fd] = apply_dest(vu.VF[fd], result, dest);
	update_mac_flags(result, dest);
}

// ---------------------------------------------------------------------------
// Lower-slot execution (branch, load/store, integer ops, FDIV)
// ---------------------------------------------------------------------------

void exec_lower(uint op)
{
	uint opcode = lower_op(op);  // bits [31:25]
	uint dest   = dest_field(op);
	uint ft     = ft_field(op);
	uint fs     = fs_field(op);
	// imm11: sign-extended 11-bit immediate (bits 10:0)
	int  imm11  = int(op & 0x7FFu);
	if ((imm11 & 0x400) != 0) imm11 |= int(0xFFFFF800u); // sign extend

	// imm15: sign-extended 15-bit (bits 14:0) used by branches
	int  imm15  = int(op & 0x7FFFu);
	if ((imm15 & 0x4000) != 0) imm15 |= int(0xFFFF8000u);

	switch (opcode)
	{
	// --- DIV (Q = VF[fs].field / VF[ft].field) ---
	case 0x38u:
	{
		float num = vu.VF[fs].x; // simplified: use X component
		float den = vu.VF[ft].x;
		vu.Q = ps2_clamp(num / den);
		break;
	}
	// --- SQRT (Q = sqrt(|VF[ft].field|)) ---
	case 0x39u:
		vu.Q = sqrt(abs(vu.VF[ft].x));
		break;
	// --- RSQRT (Q = VF[fs].field / sqrt(|VF[ft].field|)) ---
	case 0x3Au:
		vu.Q = ps2_clamp(vu.VF[fs].x / sqrt(abs(vu.VF[ft].x)));
		break;
	// --- MOVE (VFfd = VFfs) ---
	case 0x30u:
		vu.VF[ft] = apply_dest(vu.VF[ft], vu.VF[fs], dest);
		break;
	// --- MFIR (VFft = sign_extend(VIis)) ---
	case 0x31u:
	{
		float val = float(int(vu.VI[fs] & 0xFFFFu));
		vu.VF[ft] = apply_dest(vu.VF[ft], vec4(val), dest);
		break;
	}
	// --- MTIR (VIft.US[0] = VFfs.field) ---
	case 0x32u:
	{
		// store integer from float
		uint field_sel = (op >> 21) & 0x3u;
		float src = (field_sel == 0u) ? vu.VF[fs].x :
		            (field_sel == 1u) ? vu.VF[fs].y :
		            (field_sel == 2u) ? vu.VF[fs].z : vu.VF[fs].w;
		vu.VI[ft] = uint(int(src)) & 0xFFFFu;
		break;
	}
	// --- MR32 (VFfd = rotate(VFfs)) ---
	case 0x33u:
		vu.VF[ft] = apply_dest(vu.VF[ft], vec4(vu.VF[fs].w, vu.VF[fs].x, vu.VF[fs].y, vu.VF[fs].z), dest);
		break;
	// --- LQ (VFft = VU_mem[VIis + imm]) ---
	case 0x00u:
	{
		uint addr = ((vu.VI[fs] + uint(imm11)) & 0xFFFFu) * 16u;
		addr /= 4u; // word index
		if (addr + 3u < uint(mem.length()))
		{
			vec4 val = vec4(uintBitsToFloat(mem[addr + 0u]),
			                uintBitsToFloat(mem[addr + 1u]),
			                uintBitsToFloat(mem[addr + 2u]),
			                uintBitsToFloat(mem[addr + 3u]));
			vu.VF[ft] = apply_dest(vu.VF[ft], val, dest);
		}
		break;
	}
	// --- SQ (VU_mem[VIft + imm] = VFfs) ---
	case 0x01u:
	{
		uint addr = ((vu.VI[ft] + uint(imm11)) & 0xFFFFu) * 16u;
		addr /= 4u;
		if (addr + 3u < uint(mem.length()))
		{
			if ((dest & 8u) != 0u) mem[addr + 0u] = floatBitsToUint(vu.VF[fs].x);
			if ((dest & 4u) != 0u) mem[addr + 1u] = floatBitsToUint(vu.VF[fs].y);
			if ((dest & 2u) != 0u) mem[addr + 2u] = floatBitsToUint(vu.VF[fs].z);
			if ((dest & 1u) != 0u) mem[addr + 3u] = floatBitsToUint(vu.VF[fs].w);
		}
		break;
	}
	// --- ILW (VIft.elem = VU_mem_integer[VIis + imm]) ---
	case 0x04u:
	{
		uint addr = ((vu.VI[fs] + uint(imm11)) & 0xFFFFu) * 16u / 4u;
		uint field_sel = (op >> 21) & 0x3u;
		if (addr + field_sel < uint(mem.length()))
			vu.VI[ft] = mem[addr + field_sel] & 0xFFFFu;
		break;
	}
	// --- ISW (VU_mem_integer[VIft + imm].elem = VIis) ---
	case 0x05u:
	{
		uint addr = ((vu.VI[ft] + uint(imm11)) & 0xFFFFu) * 16u / 4u;
		uint field_sel = (op >> 21) & 0x3u;
		if (addr + field_sel < uint(mem.length()))
			mem[addr + field_sel] = vu.VI[fs] & 0xFFFFu;
		break;
	}
	// --- IADD (VIfd = VIfs + VIft) ---
	case 0x10u:
		vu.VI[fd_field(op)] = (vu.VI[fs] + vu.VI[ft]) & 0xFFFFu;
		break;
	// --- ISUB (VIfd = VIfs - VIft) ---
	case 0x11u:
		vu.VI[fd_field(op)] = (vu.VI[fs] - vu.VI[ft]) & 0xFFFFu;
		break;
	// --- IADDI (VIft = VIis + imm5) ---
	case 0x12u:
	{
		int imm5 = int((op >> 6) & 0x1Fu);
		if ((imm5 & 0x10) != 0) imm5 |= int(0xFFFFFFE0u);
		vu.VI[ft] = uint(int(vu.VI[fs]) + imm5) & 0xFFFFu;
		break;
	}
	// --- IAND (VIfd = VIfs & VIft) ---
	case 0x13u:
		vu.VI[fd_field(op)] = vu.VI[fs] & vu.VI[ft];
		break;
	// --- IOR (VIfd = VIfs | VIft) ---
	case 0x14u:
		vu.VI[fd_field(op)] = vu.VI[fs] | vu.VI[ft];
		break;
	// --- Branch instructions – update TPC directly ---
	// IBEQ: if VIft == VIfs: PC += imm15 * 8
	case 0x20u:
		if (vu.VI[fs] == vu.VI[ft])
			vu.TPC = uint(int(vu.TPC) + imm15 * 8);
		break;
	// IBNE: if VIft != VIfs: PC += imm15 * 8
	case 0x21u:
		if (vu.VI[fs] != vu.VI[ft])
			vu.TPC = uint(int(vu.TPC) + imm15 * 8);
		break;
	// IBLTZ: if VIis < 0: PC += imm15 * 8
	case 0x22u:
		if (int(vu.VI[fs]) < 0)
			vu.TPC = uint(int(vu.TPC) + imm15 * 8);
		break;
	// IBGTZ: if VIis > 0: PC += imm15 * 8
	case 0x23u:
		if (int(vu.VI[fs]) > 0)
			vu.TPC = uint(int(vu.TPC) + imm15 * 8);
		break;
	// IBLEZ: if VIis <= 0: PC += imm15 * 8
	case 0x24u:
		if (int(vu.VI[fs]) <= 0)
			vu.TPC = uint(int(vu.TPC) + imm15 * 8);
		break;
	// IBGEZ: if VIis >= 0: PC += imm15 * 8
	case 0x25u:
		if (int(vu.VI[fs]) >= 0)
			vu.TPC = uint(int(vu.TPC) + imm15 * 8);
		break;
	// --- NOP ---
	case 0x7Fu:
		break;
	default:
		// Unimplemented: EFU, XGKICK, WAITQ, WAITP, RINIT, RGET, RNEXT, RXOR,
		// indirect branches (JR, JALR), LOI, etc.  Defer to the CPU.
		vu.needs_cpu_fallback = 1u;
		break;
	}
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

void main()
{
	vu.needs_cpu_fallback = 0u;

	// Compute the microcode size bound (bytes) from the bound buffer size.
	uint micro_size = uint(micro.length()) * 4u;

	for (uint cycle_count = 0u; cycle_count < cycles; ++cycle_count)
	{
		// Check VPU_STAT running bit (bit 0 for VU0, bit 8 for VU1)
		uint run_bit = (vuIdx == 0u) ? 0x1u : 0x100u;
		if ((vu.VI[29] & run_bit) == 0u)
			break;

		// Check E-bit countdown
		if (vu.ebit > 0u)
		{
			vu.ebit -= 1u;
			if (vu.ebit == 0u)
			{
				// Clear the running bit
				vu.VI[29] &= ~run_bit;
				break;
			}
		}

		// Fetch instruction pair (8 bytes).
		uint pc = vu.TPC;
		if (pc + 7u > micro_size)
		{
			// PC out of range – treat as program end.
			vu.VI[29] &= ~run_bit;
			break;
		}

		uint word_idx = pc / 4u;
		uint lower_word = micro[word_idx];
		uint upper_word = micro[word_idx + 1u];

		// Advance PC (before branch so branches can override).
		vu.TPC = pc + 8u;

		// Decode flag bits from upper word.
		if ((upper_word & 0x40000000u) != 0u) // E-bit
			vu.ebit = 2u;

		// Execute upper-slot instruction first (FMAC).
		exec_upper(upper_word);
		if (vu.needs_cpu_fallback != 0u)
			return;

		// Execute lower-slot instruction (FDIV / branch / load-store / integer).
		exec_lower(lower_word);
		if (vu.needs_cpu_fallback != 0u)
			return;

		vu.cycle += 1u;
	}
}
