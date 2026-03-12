// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#include "Common.h"
#include "VUmicro.h"
#include "MTVU.h"
#include "GS.h"
#include "Gif_Unit.h"

#include "common/Console.h"

#include <algorithm>

BaseVUmicroCPU* CpuVU0 = nullptr;
BaseVUmicroCPU* CpuVU1 = nullptr;

// --------------------------------------------------------------------------------------
//  VUPluginRegistry implementation
// --------------------------------------------------------------------------------------

namespace VUPluginRegistry
{
	static std::vector<VUPluginDescriptor> s_plugins;

	void Register(const VUPluginDescriptor& descriptor)
	{
		// Prevent duplicate registrations.
		for (const auto& p : s_plugins)
		{
			if (p.type == descriptor.type)
				return;
		}
		s_plugins.push_back(descriptor);
	}

	const VUPluginDescriptor* Find(VUBackendType type)
	{
		for (const auto& p : s_plugins)
		{
			if (p.type == type)
				return &p;
		}
		return nullptr;
	}

	const std::vector<VUPluginDescriptor>& GetAll()
	{
		return s_plugins;
	}

	std::unique_ptr<BaseVUmicroCPU> CreateVU0(VUBackendType type)
	{
		const VUPluginDescriptor* desc = Find(type);
		if (!desc)
		{
			Console.Warning("VUPluginRegistry: unknown backend type %d - falling back to interpreter", static_cast<int>(type));
			desc = Find(VUBackendType::Interpreter);
		}
		if (desc && desc->IsAvailable())
			return desc->CreateVU0();

		// Ultimate fallback: use the built-in interpreter.
		Console.Warning("VUPluginRegistry: backend '%s' unavailable - using interpreter", desc ? desc->shortName : "?");
		const VUPluginDescriptor* interp = Find(VUBackendType::Interpreter);
		return interp ? interp->CreateVU0() : nullptr;
	}

	std::unique_ptr<BaseVUmicroCPU> CreateVU1(VUBackendType type)
	{
		const VUPluginDescriptor* desc = Find(type);
		if (!desc)
		{
			Console.Warning("VUPluginRegistry: unknown backend type %d - falling back to interpreter", static_cast<int>(type));
			desc = Find(VUBackendType::Interpreter);
		}
		if (desc && desc->IsAvailable())
			return desc->CreateVU1();

		Console.Warning("VUPluginRegistry: backend '%s' unavailable - using interpreter", desc ? desc->shortName : "?");
		const VUPluginDescriptor* interp = Find(VUBackendType::Interpreter);
		return interp ? interp->CreateVU1() : nullptr;
	}

	// Forward declarations - the GPU backend registers itself via RegisterBuiltins().
	void RegisterGpuBackend();

	void RegisterBuiltins()
	{
		// --- Interpreter ---
		Register({
			VUBackendType::Interpreter,
			"interp",
			"Software Interpreter (always available)",
			[]() -> bool { return true; },
			[]() -> std::unique_ptr<BaseVUmicroCPU> { return std::make_unique<InterpVU0>(); },
			[]() -> std::unique_ptr<BaseVUmicroCPU> { return std::make_unique<InterpVU1>(); },
		});

		// --- microVU Recompiler ---
		// NOTE: The factory functions here create fresh, unreserved instances.
		// VMManager::UpdateCPUImplementations() uses the pre-reserved static globals
		// (CpuMicroVU0 / CpuMicroVU1) directly rather than going through these factories.
		// These factories exist so that the registry is a complete catalogue of
		// available backends, but callers must call Reserve() before using
		// a recompiler instance obtained this way.
		Register({
			VUBackendType::Recompiler,
			"mVU",
			"microVU JIT Recompiler",
#if defined(_M_X86) || defined(_M_ARM64)
			[]() -> bool { return true; },
#else
			[]() -> bool { return false; },
#endif
			[]() -> std::unique_ptr<BaseVUmicroCPU> { return std::make_unique<recMicroVU0>(); },
			[]() -> std::unique_ptr<BaseVUmicroCPU> { return std::make_unique<recMicroVU1>(); },
		});

		// --- GPU Compute backend ---
		RegisterGpuBackend();
	}
} // namespace VUPluginRegistry

__inline u32 CalculateMinRunCycles(u32 cycles, bool requiresAccurateCycles)
{
	// If we're running an interlocked COP2 operation
	// run for an exact amount of cycles
	if(requiresAccurateCycles)
		return cycles;

	// Allow a minimum of 16 cycles to avoid running small blocks
	// Running a block of like 3 cycles is highly inefficient
	// so while sync isn't tight, it's okay to run ahead a little bit.
	return std::max(16U, cycles);
}

// Executes a Block based on EE delta time
void BaseVUmicroCPU::ExecuteBlock(bool startUp)
{
	const u32& stat = VU0.VI[REG_VPU_STAT].UL;
	const int test = m_Idx ? 0x100 : 1;

	if (m_Idx && THREAD_VU1)
	{
		vu1Thread.Get_MTVUChanges();
		return;
	}

	if (!(stat & test))
	{
		// VU currently flushes XGKICK on VU1 end so no need for this, yet
		/*if (m_Idx == 1 && VU1.xgkickenable)
		{
			_vuXGKICKTransfer((cpuRegs.cycle - VU1.xgkicklastcycle), false);
		}*/
		return;
	}

	if (startUp)
	{
		Execute(CalculateMinRunCycles(0, false));
	}
	else // Continue Executing
	{
		u32 cycle = m_Idx ? VU1.cycle : VU0.cycle;
		s32 delta = (s32)(u32)(cpuRegs.cycle - cycle);

		if (delta > 0)
			Execute(CalculateMinRunCycles(delta, false));
	}
}

// This function is called by VU0 Macro (COP2) after transferring some
// EE data to VU0's registers. We want to run VU0 Micro right after this
// to ensure that the register is used at the correct time.
// This fixes spinning/hanging in some games like Ratchet and Clank's Intro.
void BaseVUmicroCPU::ExecuteBlockJIT(BaseVUmicroCPU* cpu, bool interlocked)
{
	const u32& stat = VU0.VI[REG_VPU_STAT].UL;
	constexpr int test = 1;

	if (stat & test)
	{ // VU is running
		s32 delta = (s32)(u32)(cpuRegs.cycle - VU0.cycle);

		if (delta > 0)
		{
			cpu->Execute(CalculateMinRunCycles(delta, interlocked)); // Execute the time since the last call
		}
	}
}
