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
int vi_s16(uint r) { return int(vu.VI[r] << 16u) >> 16; }

// Write VI register (never writes to r==0)
void write_vi(uint r, uint val)
{
if (r != 0u)
vu.VI[r] = val & 0xFFFFu;
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
