// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#pragma once

// --------------------------------------------------------------------------------------
//  GpuVUmicro – Vulkan compute-shader VU backend
// --------------------------------------------------------------------------------------
// This backend offloads VU microcode execution to the GPU using a Vulkan compute
// pipeline.  The high-level flow for each Execute() call is:
//
//   1. Copy the current VURegs (registers + VU data memory + microcode) into a
//      host-visible staging buffer.
//   2. Issue a vkCmdDispatch that runs the vu_execute compute shader.  The shader
//      decodes and simulates VU instruction pairs until the requested cycle count
//      is exhausted or an E-bit instruction is encountered.
//   3. Wait for the compute fence and read back the updated VURegs from the result
//      buffer.
//
// When Vulkan is unavailable the backend transparently falls back to the software
// interpreter so callers never need to handle failure explicitly.
//
// Thread safety: GpuVUmicro0/1 are accessed only from the EE thread (same as all
// other VU backends).  The Vulkan objects are completely private to the backend and
// independent of the GS/rendering thread.

#include "VUmicro.h"

#include <memory>

// Forward declaration – the implementation hides the Vulkan context behind a Pimpl.
class GpuVUContext;

// --------------------------------------------------------------------------------------
//  GpuVUmicro0
// --------------------------------------------------------------------------------------
class GpuVUmicro0 final : public BaseVUmicroCPU
{
public:
	GpuVUmicro0();
	~GpuVUmicro0() override;

	const char* GetShortName() const override { return "gpuVU0"; }
	const char* GetLongName() const override  { return "VU0 GPU Compute Backend"; }

	void Shutdown() override;
	void Reset() override;
	void SetStartPC(u32 startPC) override;
	void Execute(u32 cycles) override;
	void Step() override;
	void Clear(u32 addr, u32 size) override;

private:
	/// Ensure the Vulkan context has been initialised (lazy init).
	bool EnsureContext();

	/// Shared Vulkan context.  nullptr before first use or when Vulkan is absent.
	std::shared_ptr<GpuVUContext> m_ctx;

	/// Fallback interpreter used when the GPU path is unavailable.
	InterpVU0 m_fallback;
};

// --------------------------------------------------------------------------------------
//  GpuVUmicro1
// --------------------------------------------------------------------------------------
class GpuVUmicro1 final : public BaseVUmicroCPU
{
public:
	GpuVUmicro1();
	~GpuVUmicro1() override;

	const char* GetShortName() const override { return "gpuVU1"; }
	const char* GetLongName() const override  { return "VU1 GPU Compute Backend"; }

	void Shutdown() override;
	void Reset() override;
	void SetStartPC(u32 startPC) override;
	void Execute(u32 cycles) override;
	void Step() override;
	void Clear(u32 addr, u32 size) override;
	void ResumeXGkick() override;

private:
	bool EnsureContext();

	std::shared_ptr<GpuVUContext> m_ctx;
	InterpVU1 m_fallback;
};

/// Register the GPU backend with the VUPluginRegistry.
/// Called automatically from VUPluginRegistry::RegisterBuiltins() via
/// VUPluginRegistry::RegisterGpuBackend().
void RegisterGpuVUBackend();
