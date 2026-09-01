// The LOD pipeline interface.
//
// A pipeline owns a LOD data structure, the device code that builds it, and the
// code that decides which parts of it to draw. It does NOT own the window, the
// camera, the loader, the rasterizer, EDL, the GL blit, timing or stats plumbing --
// all of that is shared, which is the entire point: it is what makes two pipelines
// comparable.
//
// WHY ONE COMPILED MODULE SET PER PIPELINE, rather than a switch inside a shared
// kernel. Two independent constraints force it, and both are worth understanding
// before trying to "simplify" this:
//
//   1. Every kernel is a cooperative launch using cg::this_grid().sync(). Two
//      pipelines' kernels cannot be composed into one launch.
//   2. The device-side bump allocator is deliberately non-atomic: every thread
//      redundantly walks the identical allocation sequence so all threads derive
//      identical pointers with zero atomics and zero broadcast. That requires
//      UNIFORM CONTROL FLOW across all threads, so an allocation can never sit
//      behind a pipeline-selected branch inside a kernel.
//
// Both push the selection point up to the module level -- which both research repos
// already do naturally. Code sharing happens instead at NVRTC compile time, via
// #include of the .cuh headers under kernels/shared. Zero runtime dispatch cost,
// and no uniform-control-flow hazard by construction.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <cuda.h>

#include "clod/HostDeviceCommon.h"

namespace clod {

class PointSource;
struct CloudMeta;

// ---------------------------------------------------------------------------

struct PipelineInfo {
	std::string id;           // "simlod", "cudalod", "flat", ...
	std::string displayName;

	// Consumes point batches incrementally across frames and can render a partial
	// tree (SimLOD). If false, build() waits for the whole cloud to be resident
	// and then builds in one shot (CudaLOD).
	bool progressive = false;
	bool needsWholeCloudResident = false;

	// Rough device bytes per input point, for the capability gate below. On a 16GB
	// card this is not academic: CudaLOD needs ~28 B/pt of slab plus 16 B/pt
	// resident input, so 350M points is ~15.4GB and simply does not fit, while
	// SimLOD streams the same file happily. That asymmetry must be reported, not
	// hidden behind a crash.
	double bytesPerPointEstimate = 0.0;
};

// Device memory this pipeline is allowed to use. Computed ONCE by the shell and
// handed unchanged to every pipeline, so memory figures are comparable.
//
// This replaces two upstream land grabs: SimLOD takes 80% of whatever is free at
// startup (main_progressive_octree.cpp:572), and CudaLOD takes a hardcoded slab.
// Neither is a budget. The first also makes runs non-reproducible, since it depends
// on what else happened to be on the GPU, and makes side-by-side comparison
// structurally impossible.
struct DeviceBudget {
	size_t vramTotal = 0;
	size_t vramFreeAtStartup = 0;
	size_t bytes = 0;  // what THIS pipeline may allocate
};

// GL colour attachment registered as a CUDA surface. Registered once per resize by
// GLInterop -- upstream re-registers it every frame.
struct RenderTargets {
	uint64_t surface = 0;  // cudaSurfaceObject_t
	int width = 0;
	int height = 0;
};

struct FrameContext {
	SharedUniforms uniforms;
	RenderTargets targets;
	CUstream stream = nullptr;
	int numSMs = 0;

	// Benchmark mode: synchronise and read CUevents immediately rather than one
	// frame late. Recorded in results so the two modes cannot be compared by
	// accident.
	bool strictTiming = false;
};

// Normalised, pipeline-agnostic readback for the stats panel and the benchmark
// harness.
//
// Each pipeline keeps its OWN device-side stats struct, as close to upstream as
// possible (SimLOD's Stats, CudaLOD's Results), and translates into this on the
// host. Rewriting those device structs to a common shape would mean editing the
// kernels being validated, which is exactly how a port silently stops reproducing
// its reference numbers.
struct PipelineStats {
	uint64_t numPoints = 0;
	uint64_t numVoxels = 0;
	uint64_t numPointsIngested = 0;

	uint32_t numNodes = 0;
	uint32_t numInner = 0;
	uint32_t numLeaves = 0;
	uint32_t maxPointsPerNode = 0;

	uint32_t numVisibleNodes = 0;
	uint64_t numVisiblePoints = 0;
	uint64_t numVisibleVoxels = 0;

	uint64_t bytesAllocated = 0;
	uint64_t bytesHighWater = 0;
	uint64_t bytesCapacity = 0;

	uint64_t samplesPerLevel[24] = {};

	// Health flags. Any of these set means the run is not a valid data point: the
	// structure was silently truncated. The GUI shows a banner; the benchmark
	// harness marks the record invalid rather than recording a suspiciously good
	// number.
	bool memCapacityReached = false;
	bool nodeCapacityReached = false;
	bool allocOverflow = false;
};

// ---------------------------------------------------------------------------

class ILodPipeline {
public:
	virtual ~ILodPipeline() = default;

	virtual PipelineInfo info() const = 0;

	// Compile device modules and register hot-reload watches. Must NOT make large
	// allocations -- those belong in allocate(), which knows the budget.
	// Returns false and fills err on failure; the pipeline is then unusable but the
	// application stays up.
	virtual bool initPrograms(std::string* err) = 0;

	// All device allocation happens here, within budget.bytes. Idempotent.
	virtual bool allocate(const CloudMeta& meta, const DeviceBudget& budget,
	                      std::string* err) = 0;

	// Free everything allocate() took. Called on switch-out: pipelines are
	// exclusively resident by default, because there is no way to hand SimLOD's
	// chunked octree to CudaLOD's contiguous-slice traversal anyway.
	virtual void release() = 0;

	// Clear the LOD structure and rewind any device-side ingest cursor. No realloc.
	virtual void reset() = 0;

	// Advance construction by one slice. Returns false when there is nothing left
	// to do. Progressive pipelines do one bounded launch per call; batch pipelines
	// no-op until the source is fully resident, then build once.
	virtual bool build(PointSource& source, const FrameContext& frame) = 0;

	// Draw. Must tolerate a partially built structure -- a progressive pipeline is
	// rendered while it is still ingesting.
	virtual void render(const FrameContext& frame) = 0;

	virtual const PipelineStats& stats() const = 0;

	// ImGui for tunables this pipeline OWNS (sampling strategy, node capacity,
	// batches per launch). Shared knobs belong to the shell so that every pipeline
	// is guaranteed to get the same value.
	virtual void gui() {}

	// --- filled by the pipeline from CUevents; read by the benchmark harness ---
	double buildDeviceMsTotal = 0.0;  // summed over ALL build launches
	double renderDeviceMsLast = 0.0;
	uint32_t buildLaunchCount = 0;
};

using PipelineFactory = std::function<std::unique_ptr<ILodPipeline>()>;

}  // namespace clod
