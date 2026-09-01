// Common prelude for every ClodGen kernel module.
//
// Include this first from any .cu under kernels/. It pulls in the host/device
// contract, cooperative groups, and the grid-stride helper, and brings the clod
// namespace into scope so kernel code can write `Point` rather than `clod::Point`.
//
// Reminder about the compile environment, because it surprises everyone: these
// files are compiled by NVRTC with -default-device, so unannotated functions are
// implicitly __device__ and the sources carry no __device__ markers at all. They
// read like plain C++ and will NOT compile under nvcc as-is. That is inherited from
// upstream and kept, because it genuinely is nicer to author.

#pragma once

#include <cooperative_groups.h>

#include "clod/HostDeviceCommon.h"

namespace cg = cooperative_groups;

// Deliberately NO `using clod::Point;` and friends at global scope.
//
// A ported pipeline may bring its own type of the same name: CudaLOD's device headers
// define a global `Point` and `vec3` of their own, and hoisting ours alongside them is an
// ambiguity error at the first use. Since those types are layout-identical, the port
// works fine by qualifying -- but only if this header stays out of the global namespace.
//
// So: a kernel that owns its whole translation unit says `using namespace clod;` (see
// kernels/flat/flat_render.cu); one sharing scope with foreign headers qualifies or
// imports selectively (see kernels/cudalod/cudalod_render.cu).

namespace clod {

// ---------------------------------------------------------------------------
// Grid-stride iteration
//
// Note this is a block-contiguous partition, not a classic grid-stride loop: each
// thread handles a run of consecutive indices. That is coalescing-hostile for
// streaming reads but harmless for the atomic-heavy traversal workloads it was
// written for, and it is what upstream's numbers were measured with -- so it is
// kept as-is to avoid changing the thing being compared. Use processRangeStrided
// for genuinely bandwidth-bound passes.
// ---------------------------------------------------------------------------
template <typename Fn>
void processRange(uint64_t first, uint64_t last, Fn&& fn) {
	auto grid = cg::this_grid();
	const uint64_t count = last > first ? last - first : 0ull;
	const uint64_t threads = grid.num_threads();
	const uint64_t perThread = (count + threads - 1ull) / threads;
	const uint64_t start = first + grid.thread_rank() * perThread;
	const uint64_t end = start + perThread < last ? start + perThread : last;
	for (uint64_t i = start; i < end; ++i) fn(i);
}

template <typename Fn>
void processRange(uint64_t count, Fn&& fn) {
	processRange(0ull, count, fn);
}

// Coalesced variant: consecutive threads touch consecutive elements. Prefer this
// for anything that streams points.
template <typename Fn>
void processRangeStrided(uint64_t count, Fn&& fn) {
	auto grid = cg::this_grid();
	for (uint64_t i = grid.thread_rank(); i < count; i += grid.num_threads()) {
		fn(i);
	}
}

// Nanosecond clock, for the device-side time budgets that make progressive
// construction bound its own frame cost.
inline uint64_t clodNanotime() {
	uint64_t ns;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
	return ns;
}

// ---------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------
inline uint32_t clodPackRGBA(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
	return r | (g << 8) | (b << 16) | (a << 24);
}

// Cheap integer hash, for colour-by-node debug shading.
inline uint32_t clodHashColor(uint64_t key) {
	uint64_t h = key * 0x9E3779B97F4A7C15ull;
	h ^= h >> 29;
	h *= 0xBF58476D1CE4E5B9ull;
	h ^= h >> 32;
	return clodPackRGBA(static_cast<uint32_t>(h) & 0xFFu,
	                    static_cast<uint32_t>(h >> 8) & 0xFFu,
	                    static_cast<uint32_t>(h >> 16) & 0xFFu, 255u);
}

}  // namespace clod
