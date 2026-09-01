#include "pipelines/SimlodPipeline.h"

#include <imgui.h>
#include <vector_functions.h>
#include <vector_types.h>

#include <algorithm>
#include <cstring>

#include "clod/CudaCheck.h"
#include "clod/CudaContext.h"
#include "clod/PointSource.h"
#include "clod/unsuck.hpp"  // now()

// SimLOD's own host/device contract, vendored unmodified. Deliberately NOT merged into
// clod/HostDeviceCommon.h: rewriting the struct the reference kernels read is how a port
// silently stops reproducing its published numbers.
#include "../../kernels/simlod/HostDeviceInterface.h"
// The device-layout numbers the host needs. NOT structures.cuh: that calls dot() from
// helper_math.h and only compiles inside a CUDA translation unit.
#include "../../kernels/simlod/simlod_layout.h"

namespace clod {
namespace {

// Momentary (per-launch) scratch for construction.
//
// Upstream uses 300MB, and its kernel allocates roughly 409MB from it -- 10M voxel-backlog
// entries at 16B, 10M targets at 8B, 10M spilled points at 16B, a 1M-entry chunk queue and
// assorted counters. It survives only because those arrays are never filled anywhere near
// capacity, with chunkQueue's base pointer landing entirely outside the allocation and its
// writes going into whatever cuMemAlloc returned next.
//
// Sized here from the actual allocation sum with headroom, so the buffer genuinely contains
// what the kernel hands out. Do not lower it to upstream's 300MB.
constexpr uint64_t kMomentaryBytes = 512ull << 20;

// The Node pool, from the shared layout header (static_asserted against the real struct).
constexpr uint32_t kMaxNodes = simlod::kMaxNodes;

// Persistent octree store, per input point. Chunks are 16KB for 1000 points (~16 B/pt), and
// every inner node also carries a 256KB occupancy grid, which dominates for a deep tree.
constexpr double kBytesPerPointPersistent = 48.0;
constexpr uint64_t kMinPersistentBytes = 512ull << 20;

constexpr uint64_t kBytesPerPixelScratch = 64;
constexpr uint64_t kMinScratchBytes = 64ull << 20;

float eventMs(CUevent start, CUevent end) {
	float ms = 0.0f;
	if (cuEventElapsedTime(&ms, start, end) != CUDA_SUCCESS) return 0.0f;
	return ms;
}

}  // namespace

SimlodPipeline::SimlodPipeline(CudaContext& cuda) : m_cuda(cuda) {}
SimlodPipeline::~SimlodPipeline() { release(); }

PipelineInfo SimlodPipeline::info() const {
	PipelineInfo info;
	info.id = "simlod";
	info.displayName = "SimLOD (progressive)";
	info.progressive = true;
	// False in principle -- the whole point is that it streams. True for now, because ingest
	// is the resident path until the pinned-pool loader lands.
	info.needsWholeCloudResident = true;
	// Persistent store plus the resident input points the source holds.
	info.bytesPerPointEstimate = kBytesPerPointPersistent + 16.0;
	return info;
}

bool SimlodPipeline::initPrograms(std::string* err) {
	struct Spec {
		std::unique_ptr<CudaModularProgram>* target;
		const char* module;
		const char* kernel;
	};
	const Spec specs[] = {
		{&m_resetProgram, "simlod/reset.cu", "kernel"},
		{&m_constructProgram, "simlod/progressive_octree_voxels.cu", "kernel_construct"},
		{&m_renderProgram, "simlod/simlod_render.cu", "kernel_render"},
	};

	for (const Spec& spec : specs) {
		KernelProgramDesc desc;
		desc.modules = {spec.module};
		desc.kernels = {spec.kernel};
		*spec.target = std::make_unique<CudaModularProgram>(std::move(desc));
		if (!(*spec.target)->ok()) {
			if (err) *err = (*spec.target)->lastError();
			return false;
		}
	}

	// A tree built by the previous version of the construct kernel is not valid input to the
	// new one, so a hot reload must reset. Upstream does not do this and leaves the old tree
	// on screen, which is a confusing thing to debug.
	m_constructProgram->onCompile([this] { m_needsReset = true; });

	return true;
}

bool SimlodPipeline::allocate(const CloudMeta& meta, const DeviceBudget& budget,
                              std::string* err) {
	release();

	m_stats = PipelineStats{};
	m_stats.bytesCapacity = budget.bytes;
	m_numPoints = meta.numPoints;

	if (meta.numPoints == 0) {
		if (err) *err = "cloud is empty";
		return false;
	}

	const uint64_t inputBytes = meta.numPoints * 16ull;
	const uint64_t available =
		budget.bytes > inputBytes ? budget.bytes - inputBytes : 0;

	m_nodesBytes = static_cast<uint64_t>(kMaxNodes) * simlod::kNodeBytes;
	m_momentaryBytes = kMomentaryBytes;

	uint64_t wantPersistent = static_cast<uint64_t>(
		kBytesPerPointPersistent * static_cast<double>(meta.numPoints));
	wantPersistent = std::max(wantPersistent, kMinPersistentBytes);

	const uint64_t fixed = m_nodesBytes + m_momentaryBytes;
	if (available < fixed + kMinPersistentBytes) {
		if (err) {
			*err = "not enough device memory: needs at least " +
			       std::to_string((fixed + kMinPersistentBytes) / (1024 * 1024)) +
			       " MB, " + std::to_string(available / (1024 * 1024)) + " MB available";
		}
		return false;
	}
	m_persistentBytes = std::min(wantPersistent, available - fixed);

	auto alloc = [&](CUdeviceptr* ptr, uint64_t bytes, const char* what) {
		if (CLOD_CU(cuMemAlloc(ptr, bytes)) != CUDA_SUCCESS) {
			if (err) {
				*err = std::string("cuMemAlloc failed for ") + what + " (" +
				       std::to_string(bytes / (1024 * 1024)) + " MB)";
			}
			return false;
		}
		CLOD_CU(cuMemsetD8(*ptr, 0, bytes));
		return true;
	};

	if (!alloc(&m_momentary, m_momentaryBytes, "the momentary buffer") ||
	    !alloc(&m_persistent, m_persistentBytes, "the persistent octree store") ||
	    !alloc(&m_nodes, m_nodesBytes, "the node pool") ||
	    !alloc(&m_statsBuffer, sizeof(Stats), "stats") ||
	    !alloc(&m_frameStart, 8, "the frame timestamp") ||
	    // CudaPrint is a no-op on both ends but is threaded through both kernel
	    // signatures, so it gets a small real allocation rather than a null pointer.
	    !alloc(&m_cudaPrint, 1024, "the CudaPrint buffer") ||
	    !alloc(&m_diagnostics, sizeof(DeviceDiagnostics), "diagnostics") ||
	    // Must hold BATCH_STREAM_SIZE entries: reset writes all of them. See the member
	    // comment in the header for why reset does not get the real buffer.
	    !alloc(&m_resetBatchSizes, uint64_t(simlod::kBatchStreamSize) * 4,
	           "reset's batchSizes scratch") ||
	    !alloc(&m_resetNumUploaded, 4, "reset's numBatchesUploaded scratch")) {
		release();
		return false;
	}

	CLOD_CU(cuEventCreate(&m_buildStart, CU_EVENT_DEFAULT));
	CLOD_CU(cuEventCreate(&m_buildEnd, CU_EVENT_DEFAULT));

	m_stats.numPoints = 0;
	m_stats.bytesAllocated = m_persistentBytes + m_momentaryBytes + m_nodesBytes;
	m_needsReset = true;
	m_complete = false;
	return true;
}

void SimlodPipeline::release() {
	for (CUdeviceptr* p : {&m_momentary, &m_persistent, &m_nodes, &m_statsBuffer,
	                       &m_frameStart, &m_cudaPrint, &m_scratch, &m_diagnostics,
	                       &m_resetBatchSizes, &m_resetNumUploaded}) {
		if (*p) {
			CLOD_CU(cuMemFree(*p));
			*p = 0;
		}
	}
	m_momentaryBytes = m_persistentBytes = m_nodesBytes = m_scratchBytes = 0;

	for (CUevent* e : {&m_buildStart, &m_buildEnd}) {
		if (*e) {
			cuEventDestroy(*e);
			*e = nullptr;
		}
	}
	m_complete = false;
	m_needsReset = true;
}

void SimlodPipeline::reset() {
	m_needsReset = true;
	m_complete = false;
	buildDeviceMsTotal = 0.0;
	buildLaunchCount = 0;
	m_lastBuildMs = 0.0;
	m_batchesConsumed = 0;
}

void SimlodPipeline::fillUniforms(const FrameContext& frame, void* out) const {
	Uniforms* u = static_cast<Uniforms*>(out);
	std::memset(u, 0, sizeof(Uniforms));

	// Only four fields are read by kernel_construct: boxMin, boxMax, frameCounter and
	// persistentBufferCapacity. The rest exist for upstream's own renderer, which we do not
	// use -- shading is driven from SharedUniforms instead, so they stay zero rather than
	// becoming controls that appear to do something.
	u->boxMin = make_float3(frame.uniforms.boxMin.x, frame.uniforms.boxMin.y,
	                        frame.uniforms.boxMin.z);
	u->boxMax = make_float3(frame.uniforms.boxMax.x, frame.uniforms.boxMax.y,
	                        frame.uniforms.boxMax.z);
	u->frameCounter = frame.uniforms.frameCounter;
	u->persistentBufferCapacity = m_persistentBytes;
	u->momentaryBufferCapacity = m_momentaryBytes;
	u->width = frame.uniforms.width;
	u->height = frame.uniforms.height;
}

bool SimlodPipeline::build(PointSource& source, const FrameContext& frame) {
	if (!m_constructProgram || !m_constructProgram->ok()) return false;
	if (!m_resetProgram || !m_resetProgram->ok()) return false;
	if (!m_momentary || !m_persistent || !m_nodes) return false;

	const BatchView view = source.view();
	if (view.slots == 0 || view.batchSizes == 0 || view.numBatchesUploaded == 0) {
		// The source has not published a batch view. Nothing to do; not an error, since a
		// streaming source may simply not have started yet.
		return true;
	}
	m_batchesTotal = view.numBatchesTotal;

	Uniforms uniforms;
	fillUniforms(frame, &uniforms);

	CUdeviceptr persistent = m_persistent, nodes = m_nodes;
	CUdeviceptr statsPtr = m_statsBuffer;
	CUdeviceptr frameStart = m_frameStart, cudaPrint = m_cudaPrint;
	CUdeviceptr batchSizes = view.batchSizes;
	CUdeviceptr numUploaded = view.numBatchesUploaded;

	// --- reset, if needed -------------------------------------------------
	if (m_needsReset) {
		CUfunction resetKernel = m_resetProgram->kernel("kernel");
		if (!resetKernel) return false;

		// Scratch, not the source's real metadata -- see the header.
		CUdeviceptr resetSizes = m_resetBatchSizes;
		CUdeviceptr resetUploaded = m_resetNumUploaded;
		void* resetArgs[] = {&uniforms,  &persistent,    &nodes,     &statsPtr,
		                     &cudaPrint, &resetUploaded, &resetSizes};

		// One block, one thread -- but a COOPERATIVE launch, because reset.cu calls
		// grid.sync(). A plain cuLaunchKernel makes that undefined, and the failure mode is
		// a bare CUDA_ERROR_LAUNCH_FAILED ("unspecified launch failure") that says nothing
		// about the cause. Every kernel in both reference pipelines grid-syncs, so
		// cooperative launch is the rule here, not the exception.
		CLOD_CU(cuLaunchCooperativeKernel(resetKernel, 1, 1, 1, 1, 1, 1, 0, 0, resetArgs));

		const CUresult sync = cuCtxSynchronize();
		if (isStickyError(sync)) {
			reportDeadContextAndExit(sync, "SimLOD reset kernel");
		}
		CLOD_CU(sync);

		m_needsReset = false;
		m_complete = false;
		m_batchesConsumed = 0;
	}

	if (m_complete) return false;

	// --- one bounded construction step ------------------------------------
	CUfunction construct = m_constructProgram->kernel("kernel_construct");
	if (!construct) return false;

	// The device-side 10ms budget is measured against this, so it must be refreshed every
	// launch or the kernel believes it is already out of time.
	const uint64_t nowNs = static_cast<uint64_t>(now() * 1e9);
	CLOD_CU(cuMemcpyHtoD(m_frameStart, &nowNs, sizeof(nowNs)));

	CUdeviceptr points = view.slots;
	CUdeviceptr momentary = m_momentary;

	void* args[] = {&uniforms, &points,     &momentary,  &persistent, &nodes,
	                &statsPtr, &frameStart, &cudaPrint,  &numUploaded, &batchSizes};

	// Upstream launches this at exactly 1 block per SM.
	const int grid = m_cuda.gridForKernel(construct, m_blockSize, 1);

	CLOD_CU(cuEventRecord(m_buildStart, 0));
	CLOD_CU(cuLaunchCooperativeKernel(construct, static_cast<unsigned>(grid), 1, 1,
	                                  static_cast<unsigned>(m_blockSize), 1, 1, 0, 0,
	                                  args));
	CLOD_CU(cuEventRecord(m_buildEnd, 0));

	const CUresult sync = cuCtxSynchronize();
	if (isStickyError(sync)) {
		reportDeadContextAndExit(sync, "SimLOD kernel_construct");
	}
	CLOD_CU(sync);

	m_lastBuildMs = eventMs(m_buildStart, m_buildEnd);
	buildDeviceMsTotal += m_lastBuildMs;
	++buildLaunchCount;

	readStats();

	// Progressive: done once every batch has been consumed.
	if (m_batchesTotal > 0 && m_batchesConsumed >= m_batchesTotal) {
		m_complete = true;
		return false;
	}
	return true;  // more to do next frame
}

void SimlodPipeline::readStats() {
	if (!m_statsBuffer) return;

	Stats s = {};
	if (CLOD_CU(cuMemcpyDtoH(&s, m_statsBuffer, sizeof(Stats))) != CUDA_SUCCESS) return;

	m_batchesConsumed = s.batchletIndex;

	m_stats.numPoints = s.numPoints;
	m_stats.numVoxels = s.numVoxels;
	m_stats.numPointsIngested = s.numPointsProcessed;
	m_stats.numNodes = s.numNodes;
	m_stats.numInner = s.numInner;
	m_stats.numLeaves = s.numLeaves;
	m_stats.numVisibleNodes = s.numVisibleNodes;
	m_stats.numVisiblePoints = s.numVisiblePoints;
	m_stats.numVisibleVoxels = s.numVisibleVoxels;

	m_stats.bytesHighWater = s.allocatedBytes_persistent;
	m_stats.bytesAllocated = m_persistentBytes + m_momentaryBytes + m_nodesBytes;
	m_stats.memCapacityReached = s.memCapacityReached;

	// numNodes is a bump index grown by `atomicAdd(&stats->numNodes, 8)` with no capacity
	// check on the device, so this is the only place the pool running out can be noticed.
	if (m_stats.numNodes >= kMaxNodes) m_stats.nodeCapacityReached = true;
}

void SimlodPipeline::ensureScratch(int width, int height) {
	const uint64_t pixels =
		static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
	uint64_t needed = std::max(pixels * kBytesPerPixelScratch, kMinScratchBytes);
	// The render kernel also allocates the draw list plus two MAX_NODES_CAPACITY flag
	// arrays.
	needed += 16ull << 20;
	if (needed <= m_scratchBytes) return;

	if (m_scratch) CLOD_CU(cuMemFree(m_scratch));
	m_scratch = 0;
	if (CLOD_CU(cuMemAlloc(&m_scratch, needed)) != CUDA_SUCCESS) {
		m_scratchBytes = 0;
		return;
	}
	m_scratchBytes = needed;
}

void SimlodPipeline::render(const FrameContext& frame) {
	if (!m_renderProgram || !m_renderProgram->ok()) return;
	if (frame.targets.surface == 0) return;
	if (!m_nodes || m_needsReset) return;

	ensureScratch(frame.targets.width, frame.targets.height);
	if (!m_scratch) return;

	CUfunction kernel = m_renderProgram->kernel("kernel_render");
	if (!kernel) return;

	RenderArgs args;
	args.uniforms = frame.uniforms;
	args.scratch = reinterpret_cast<uint32_t*>(m_scratch);
	args.scratchCapacity = m_scratchBytes;
	args.surface = frame.targets.surface;

	CUdeviceptr nodes = m_nodes, statsPtr = m_statsBuffer, diag = m_diagnostics;
	void* kernelArgs[] = {&args, &nodes, &statsPtr, &diag};

	const int grid = m_cuda.gridForKernel(kernel, m_blockSize);
	CLOD_CU(cuLaunchCooperativeKernel(kernel, static_cast<unsigned>(grid), 1, 1,
	                                  static_cast<unsigned>(m_blockSize), 1, 1, 0, 0,
	                                  kernelArgs));

	if (frame.strictTiming) {
		const CUresult sync = cuCtxSynchronize();
		if (isStickyError(sync)) {
			reportDeadContextAndExit(sync, "SimLOD kernel_render");
		}
		CLOD_CU(sync);
	}

	DeviceDiagnostics d = {};
	if (cuMemcpyDtoH(&d, m_diagnostics, sizeof(DeviceDiagnostics)) == CUDA_SUCCESS) {
		if (d.allocOverflow) m_stats.allocOverflow = true;
		// Our render kernel owns selection, so these come from it rather than from
		// upstream's Stats (which our renderer never writes).
		m_stats.numVisibleNodes = d.drawItems;
		m_stats.numVisiblePoints = d.drawSamples;
	}
}

void SimlodPipeline::gui() {
	ImGui::TextUnformatted(
		"Progressive construction: one bounded launch per frame inserts\n"
		"batches into the octree while it is being rendered.");
	ImGui::Separator();

	if (m_batchesTotal > 0) {
		const float progress =
			static_cast<float>(m_batchesConsumed) / static_cast<float>(m_batchesTotal);
		char overlay[64];
		snprintf(overlay, sizeof(overlay), "%u / %u batches", m_batchesConsumed,
		         m_batchesTotal);
		ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), overlay);
	}

	if (ImGui::BeginTable("simlod_timing", 2, ImGuiTableFlags_SizingStretchProp)) {
		auto row = [](const char* label, const char* fmt, double v) {
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			ImGui::TextUnformatted(label);
			ImGui::TableNextColumn();
			ImGui::Text(fmt, v);
		};
		row("last launch", "%.2f ms", m_lastBuildMs);
		row("build total", "%.2f ms", buildDeviceMsTotal);
		row("launches", "%.0f", static_cast<double>(buildLaunchCount));
		const double mps = buildDeviceMsTotal > 0.0
		                       ? double(m_stats.numPointsIngested) / 1e6 /
		                             (buildDeviceMsTotal / 1000.0)
		                       : 0.0;
		row("throughput", "%.0f MP/s", mps);
		row("persistent", "%.2f GB", double(m_persistentBytes) / 1e9);
		row("high water", "%.2f GB", double(m_stats.bytesHighWater) / 1e9);
		ImGui::EndTable();
	}

	if (ImGui::Button("rebuild")) m_needsReset = true;

	ImGui::TextDisabled(
		"ingest is the resident path, not the paper's overlapped\n"
		"streaming loader -- construction is progressive, loading is not");

	if (m_renderProgram && m_renderProgram->isStale()) {
		ImGui::TextColored(ImVec4(1, 0.4f, 0.2f, 1),
		                   "render kernel failed to recompile;\n"
		                   "showing the previously loaded version");
	}
}

}  // namespace clod
