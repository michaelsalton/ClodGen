// The few SimLOD device-layout numbers the HOST needs, in a header the host can compile.
//
// structures.cuh cannot be included host-side: its Node methods call dot() from
// helper_math.h, so it only compiles inside a CUDA translation unit. Rather than dragging
// the CUDA sample headers into the host build, the two numbers the host actually needs to
// size allocations live here, and structures.cuh static_asserts that they still match the
// real types. Drift becomes a compile error in the kernel rather than a silent mis-sized
// node pool.
#pragma once

#include <cstdint>

namespace clod::simlod {

// Node pool capacity. Upstream grows the pool with `atomicAdd(&stats->numNodes, 8)` and no
// capacity check anywhere, so this is also the bound both kernels clamp against.
constexpr uint32_t kMaxNodes = 200'000;

// sizeof(Node). Verified against the real struct by a static_assert in structures.cuh.
constexpr uint32_t kNodeBytes = 152;

// Must equal BATCH_STREAM_SIZE in structures.cuh (static_asserted there).
//
// The host needs it because the reset kernel writes batchSizes[0 .. BATCH_STREAM_SIZE-1]
// unconditionally, so any buffer handed to it must be at least that large. Getting this
// wrong is an out-of-bounds device write, which presents as CUDA_ERROR_LAUNCH_FAILED with
// no indication of where.
constexpr uint32_t kBatchStreamSize = 8192;

}  // namespace clod::simlod
