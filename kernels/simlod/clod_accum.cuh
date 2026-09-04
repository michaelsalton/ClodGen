// ClodGen's own file. NOT vendored -- it carries no upstream header, unlike its
// neighbours in this directory, and nothing in it needs to stay byte-identical to
// anything.
//
// The accumulator subpass: the seventh phase of addBatch(), and the first half of the
// detail-aware work (plans/03_AccumulatorHook.md). It maintains per-node running sums so
// the Analysis kernel has something to roll up. It computes nothing, decides nothing, and
// -- this is the load-bearing part -- MUTATES NO TREE STATE. It writes only to the side
// array in clod/SimlodAccum.h. That is what makes "--dump-frame must stay byte-identical"
// the acceptance test.
//
// WHY IT IS A SEPARATE PASS RATHER THAN AN EDIT TO insertPoints
//
// CLAUDE.md permits two ways to hook vendored device code: a separate pass that
// re-traverses the batch, or an #ifdef-guarded macro so the default build emits identical
// device code. This is the first, because the accumulator is not instrumentation -- it is
// the feature, and a build variant that is off by default is a feature that never runs.
// The price is one extra octree descent per point per batch, which is measured rather
// than assumed; §3.4 of the plan lists the two fallbacks if it turns out to be too much.
//
// WHY STEP 4 AND NOT STEP 1
//
// Step 1's counting sub-pass already tallies, but it runs once per expansion iteration
// (up to 20), has its own don't-count-twice logic, and re-feeds the spill buffer from
// scratch. Step 4 runs exactly once per point per update, at the single node where the
// point comes to rest.
//
// INCLUDE ORDER: this header uses Node, Point, MAX_DEPTH and processRange, so it must
// come after utils.h.cu and structures.cuh in the including translation unit.
#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "clod/SimlodAccum.h"

namespace cg = cooperative_groups;

using clod::AccumGlobals;
using clod::NodeAccum;

// Warp-aggregate the atomics, or issue one set per point.
//
// false is the Stage 2 reference implementation from the plan -- the simplest correct
// version, and what the aggregated path is validated against (`count` must be
// bit-identical, the sums within float tolerance). Kept compiling rather than deleted so
// that check can be re-run after any change here; the branch is uniform and constant, so
// the unused path costs nothing at runtime.
constexpr bool CLOD_ACCUM_AGGREGATE = true;

// ---------------------------------------------------------------------------
// Morton
// ---------------------------------------------------------------------------

// Spread the low 20 bits of v out with two zero bits between each, so three of these can
// be interleaved into a 60-bit code.
inline uint64_t clodMortonSpread(uint32_t v) {
	uint64_t x = uint64_t(v) & 0xFFFFFull;
	x = (x | (x << 32)) & 0x001F00000000FFFFull;
	x = (x | (x << 16)) & 0x001F0000FF0000FFull;
	x = (x | (x <<  8)) & 0x100F00F00F00F00Full;
	x = (x | (x <<  4)) & 0x10C30C30C30C30C3ull;
	x = (x | (x <<  2)) & 0x1249249249249249ull;
	return x;
}

// X MOST SIGNIFICANT, deliberately: it has to match
//   childIndex = (child_X << 2) | (child_Y << 1) | child_Z
// in the traversals in this file. Interleaved the other way round, Morton order is not
// the octree's own child order and the Analysis kernel's closure test would be comparing
// incomparable things.
//
// The watermark this feeds is well-defined and cheap, but it is NOT YET A CLOSURE ORACLE:
// nothing in ClodGen sorts points into Morton order (every reader hands them over in file
// order, which for .las/.laz is acquisition order along flight lines). Building closure on
// it needs either an ordering stage in the loader or a different criterion -- an Analysis
// decision. This layer only guarantees the watermark is maintained and correctly oriented.
inline uint64_t clodMortonCode(uint32_t X, uint32_t Y, uint32_t Z) {
	return (clodMortonSpread(X) << 2) | (clodMortonSpread(Y) << 1) | clodMortonSpread(Z);
}

// ---------------------------------------------------------------------------
// Clear on split
// ---------------------------------------------------------------------------

inline void clodAccumClear(NodeAccum* a) {
	for (int i = 0; i < clod::kNumGeomSums; i++) a->g[i] = 0.0;
	for (int i = 0; i < clod::kNumColorSums; i++) a->c[i] = 0.0f;
	a->count = 0;
	a->lastTouchedBatch = 0;
	a->state = clod::kAccumOpen;
	a->score = 0.0f;
}

// A leaf that spilled during step 1 is now an inner node, and its sums are meaningless:
// inner-node statistics are derived by roll-up later, not stored. Its points were moved
// to spilledPoints and are re-inserted -- and re-accumulated -- into the new leaves in
// this same update, so there is no double-counting to reason about.
//
// This is done here rather than at the split site for three reasons. It keeps
// doSplitting() untouched, which keeps the vendored diff to one call. It is idempotent
// and self-healing, so a clear that is somehow missed is corrected on the next batch
// instead of silently poisoning a roll-up. And it makes the invariant observable: the
// stats pass counts inner nodes that still hold sums, which must be zero.
//
// Cost is one pass over at most MAX_NODES_CAPACITY nodes per batch, against ~1M points of
// real work.
void clodAccumClearInner(Node* nodes, NodeAccum* accums, uint32_t numNodes) {
	uint32_t count = min(numNodes, MAX_NODES_CAPACITY);

	processRange(count, [&](int nodeIndex) {
		Node* node = &nodes[nodeIndex];
		if (node->isLeafFn()) return;
		if (accums[nodeIndex].count == 0) return;

		clodAccumClear(&accums[nodeIndex]);
	});
}

// ---------------------------------------------------------------------------
// Accumulation
// ---------------------------------------------------------------------------

void clodAccumPoint(
	Node* root, Point point,
	Node* nodes, NodeAccum* accums, AccumGlobals* globals,
	float3 octreeMin, float octreeSize,
	uint32_t batchIndex
) {
	// --- descend to the leaf --------------------------------------------------
	//
	// Byte-for-byte the traversal insertPoints() performs, including the float
	// arithmetic. It has to be: a point must accumulate at the SAME leaf it was inserted
	// into, or `sumLeafCounts == Stats::numPoints` fails, and a subtly different
	// expression here would land a boundary point one cell over.
	float fGridSize = pow(2.0f, float(MAX_DEPTH));

	uint32_t X = fGridSize * (point.x - octreeMin.x) / octreeSize;
	uint32_t Y = fGridSize * (point.y - octreeMin.y) / octreeSize;
	uint32_t Z = fGridSize * (point.z - octreeMin.z) / octreeSize;

	Node* current = root;

	for (int level = 0; level < MAX_DEPTH; level++) {
		uint32_t level_X = X >> (MAX_DEPTH - level - 1);
		uint32_t level_Y = Y >> (MAX_DEPTH - level - 1);
		uint32_t level_Z = Z >> (MAX_DEPTH - level - 1);

		uint32_t child_X = level_X & 1;
		uint32_t child_Y = level_Y & 1;
		uint32_t child_Z = level_Z & 1;

		uint32_t childIndex = (child_X << 2) | (child_Y << 1) | child_Z;

		if (current->children[childIndex] == nullptr) break;

		current = current->children[childIndex];
	}

	Node* leaf = current;

	// The node pool is a bump index with NO device-side capacity check -- the host
	// clamping in SimlodPipeline::readStats is the only place exhaustion is noticed. So
	// bound the index before using it, rather than writing past the side array.
	uint64_t nodeIndex = uint64_t(leaf - nodes);
	if (nodeIndex >= MAX_NODES_CAPACITY) return;

	// --- node-local coordinates, in double ------------------------------------
	//
	// THE SINGLE MOST IMPORTANT DETAIL IN THE HOOK. Sum raw world coordinates -- UTM
	// eastings in the hundreds of thousands -- and Sxx loses precisely the small
	// differences the smallest eigenvalue is made of.
	//
	// Computed in double because the left term is a cancellation at depth: at level 15 it
	// is ~32768 and the answer is in [0,1], which float32 resolves to about 2e-3 of a
	// node. A double reciprocal multiply costs single-digit microseconds over a 36M-point
	// cloud.
	//
	// NOT reusing the integer X,Y,Z the traversal above already computed: those quantise
	// the point to 2^-20 of the ROOT cube (1.3 mm on morro_bay), which is a floor on the
	// smallest eigenvalue that shallow nodes cannot see past.
	double invOctreeSize = 1.0 / double(octreeSize);
	double levelScale = double(1ull << leaf->level);

	double lx = (double(point.x) - double(octreeMin.x)) * invOctreeSize * levelScale - double(leaf->X);
	double ly = (double(point.y) - double(octreeMin.y)) * invOctreeSize * levelScale - double(leaf->Y);
	double lz = (double(point.z) - double(octreeMin.z)) * invOctreeSize * levelScale - double(leaf->Z);

	// The float->uint32 truncation that picked the cell and this normalisation are not
	// the same arithmetic, so a point exactly on a cell boundary can land marginally
	// outside. One instruction each, and it keeps a garbage term out of a sum that is
	// never recomputed.
	lx = fmin(fmax(lx, 0.0), 1.0);
	ly = fmin(fmax(ly, 0.0), 1.0);
	lz = fmin(fmax(lz, 0.0), 1.0);

	double g[clod::kNumGeomSums];
	g[clod::kSumX]  = lx;
	g[clod::kSumY]  = ly;
	g[clod::kSumZ]  = lz;
	g[clod::kSumXX] = lx * lx;
	g[clod::kSumXY] = lx * ly;
	g[clod::kSumXZ] = lx * lz;
	g[clod::kSumYY] = ly * ly;
	g[clod::kSumYZ] = ly * lz;
	g[clod::kSumZZ] = lz * lz;

	// RGBA8, alpha in the high byte. Channels in [0,1] so the colour sums are scale-free
	// in the same way the geometry ones are.
	float cr = float((point.color >>  0) & 0xFFu) * (1.0f / 255.0f);
	float cg_ = float((point.color >>  8) & 0xFFu) * (1.0f / 255.0f);
	float cb = float((point.color >> 16) & 0xFFu) * (1.0f / 255.0f);

	float c[clod::kNumColorSums];
	c[clod::kSumR]  = cr;
	c[clod::kSumG]  = cg_;
	c[clod::kSumB]  = cb;
	c[clod::kSumRR] = cr * cr;
	c[clod::kSumGG] = cg_ * cg_;
	c[clod::kSumBB] = cb * cb;

	uint64_t morton = clodMortonCode(X, Y, Z);

	NodeAccum* a = &accums[nodeIndex];

	if (CLOD_ACCUM_AGGREGATE) {
		// The idiom upstream already wrote in this file for exactly this operation
		// (progressive_octree_voxels.cu:215-221). Do NOT hand-roll a butterfly reduction
		// over a __match_any_sync mask instead: a guarded __shfl_xor_sync over an
		// arbitrary lane subset does not converge to the group total -- for the mask
		// {lane 0, lane 3} every partner is outside the mask at every step, so the total
		// exists nowhere. That produces plausible-but-wrong sums rather than a crash, and
		// any warp that is not fully converged can produce such a mask.
		//
		// labeled_partition is built on coalesced_threads(), so it is correct for whatever
		// subset of the warp is actually converged.
		auto warp = cg::coalesced_threads();
		auto group = cg::labeled_partition(warp, nodeIndex);

		if (group.num_threads() == 1) {
			// Early out. Without it, the all-distinct-leaves case pays ~75 shuffles to
			// save nothing -- and with unsorted input (see clodMortonCode above) that case
			// is not rare.
			for (int i = 0; i < clod::kNumGeomSums; i++) atomicAdd(&a->g[i], g[i]);
			for (int i = 0; i < clod::kNumColorSums; i++) atomicAdd(&a->c[i], c[i]);
			atomicAdd(&a->count, 1u);
			a->lastTouchedBatch = batchIndex;

			atomicMax(&globals->mortonWatermark, morton);
			atomicAdd(&globals->numAccumulated, 1ull);
			atomicAdd(&globals->numAtomicGroups, 1ull);
		} else {
			double sg[clod::kNumGeomSums];
			for (int i = 0; i < clod::kNumGeomSums; i++) {
				sg[i] = cg::reduce(group, g[i], cg::plus<double>());
			}
			// The partial sums INSIDE a warp reduction can stay float: at most 32 terms,
			// so this contributes ~1e-5 absolute against the 0.05 that matters.
			float sc[clod::kNumColorSums];
			for (int i = 0; i < clod::kNumColorSums; i++) {
				sc[i] = cg::reduce(group, c[i], cg::plus<float>());
			}
			// One atomicMax per point on a single global address would be 36M serialised
			// atomics on one cache line.
			uint64_t groupMorton = cg::reduce(group, morton, cg::greater<uint64_t>());

			if (group.thread_rank() == 0) {
				for (int i = 0; i < clod::kNumGeomSums; i++) atomicAdd(&a->g[i], sg[i]);
				for (int i = 0; i < clod::kNumColorSums; i++) atomicAdd(&a->c[i], sc[i]);
				atomicAdd(&a->count, group.num_threads());
				// Plain store: every writer writes the same value, since batchIndex is
				// uniform across the launch's current batch.
				a->lastTouchedBatch = batchIndex;

				atomicMax(&globals->mortonWatermark, groupMorton);
				atomicAdd(&globals->numAccumulated, (unsigned long long)group.num_threads());
				atomicAdd(&globals->numAtomicGroups, 1ull);
			}
		}
	} else {
		// Stage 2 reference: one set of atomics per point.
		for (int i = 0; i < clod::kNumGeomSums; i++) atomicAdd(&a->g[i], g[i]);
		for (int i = 0; i < clod::kNumColorSums; i++) atomicAdd(&a->c[i], c[i]);
		atomicAdd(&a->count, 1u);
		a->lastTouchedBatch = batchIndex;

		atomicMax(&globals->mortonWatermark, morton);
		atomicAdd(&globals->numAccumulated, 1ull);
		atomicAdd(&globals->numAtomicGroups, 1ull);
	}
}

// The subpass. Runs after insertVoxels, which is later than its true dependency (expand)
// -- it reads only the tree topology and the two point arrays. Last, so the existing
// t_00..t_70 phase deltas keep their meaning and stay comparable to runs already
// captured, and so fusing it into insertPoints later stays an option.
//
// Both point arrays are processed, mirroring insertPoints exactly. doCounting walks every
// point of each spilling node into spilledPoints and doSplitting then zeroes the node, so
// a freshly created child that itself spills during a later expand iteration contributes
// nothing: each point is spilled at most once per batch and re-inserted exactly once.
void clodAccumulate(
	Node* root, Point* points, uint32_t numPoints,
	Point* spilledPoints, uint32_t numSpilledPoints,
	Node* nodes, NodeAccum* accums, AccumGlobals* globals,
	uint32_t numNodes,
	float3 octreeMin, float octreeSize,
	uint32_t batchIndex
) {
	if (accums == nullptr || globals == nullptr) return;

	auto grid = cg::this_grid();

	clodAccumClearInner(nodes, accums, numNodes);

	grid.sync();

	processRange(numPoints, [&](int pointID) {
		clodAccumPoint(root, points[pointID], nodes, accums, globals,
		               octreeMin, octreeSize, batchIndex);
	});

	grid.sync();

	processRange(numSpilledPoints, [&](int pointID) {
		clodAccumPoint(root, spilledPoints[pointID], nodes, accums, globals,
		               octreeMin, octreeSize, batchIndex);
	});

	grid.sync();
}
