// The one header a pipeline's device code needs to include.
//
//   #include "shared/clod_pipeline.cuh"
//   using namespace clod;
//
// It pulls in, in dependency order: the host/device contract and grid helpers, the
// bounds-checked bump allocators, device math and frustum culling, the shared
// software rasteriser, and the octree wireframe overlay.
//
// Sharing happens HERE, at NVRTC compile time, rather than by dispatching inside a
// kernel. That is not a stylistic preference -- see the note in
// include/clod/ILodPipeline.h: cooperative launches cannot be composed, and the
// non-atomic allocator forbids branch-dependent allocation. Textual inclusion gives
// every pipeline the same rasteriser with zero runtime dispatch and no
// uniform-control-flow hazard.

#pragma once

#include "shared/clod_prelude.cuh"
#include "shared/clod_alloc.cuh"
#include "shared/clod_math.cuh"
#include "shared/clod_framebuffer.cuh"
#include "shared/clod_draw.cuh"
#include "shared/clod_lines.cuh"
