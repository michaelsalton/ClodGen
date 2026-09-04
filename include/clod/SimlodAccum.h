// Per-node geometry and colour accumulators for SimLOD's octree.
//
// This is the data layer under the detail-aware work (plans/03_AccumulatorHook.md): the
// Analysis kernel rolls these sums up the tree and turns them into scores, and the
// Refinement kernel spends budget on the result. Nothing here computes or decides
// anything -- a NodeAccum is fifteen running sums and a counter.
//
// WHY A SIDE ARRAY AND NOT FIELDS ON Node
//
// SimLOD's Node is vendored device code, kept byte-identical so the port stays verifiable
// against bench/reference/. sizeof(Node) == 152 is static_asserted in
// kernels/simlod/structures.cuh and mirrored host-side as kNodeBytes in
// kernels/simlod/simlod_layout.h to size the 200k-node pool. Widening it would mean
// editing upstream code, one static_assert and one host constant in step -- and it would
// cost the render kernel's hot traversal its 8-nodes-per-cache-line layout for data that
// traversal never reads. So the accumulator is a flat array indexed by node index
// (node - nodes), allocated by the host, and can be resized or dropped without touching
// the tree.
//
// TWO RULES THIS HEADER FOLLOWS, plus a third that is specific to SimLOD
//
//   1. Nothing host-only. NVRTC has no libstdc++.  (as HostDeviceCommon.h)
//   2. Layout identical on both sides. No virtuals, no std::, explicit padding.
//
//   3. NO INCLUDES AT ALL, AND NO IMPORTED TYPE NAMES. Unlike HostDeviceCommon.h, this
//      header is included by SimLOD's VENDORED translation unit, which gets its
//      fixed-width types from kernels/simlod/utils.h.cu -- where uint64_t is
//      `typedef unsigned long long`. CCCL's cuda::std::uint64_t is `unsigned long` on
//      Linux, a DIFFERENT type, so a `using cuda::std::uint64_t;` in scope there is a
//      conflicting-typedef error. That is the same trap patches/cudalod-linux-port.patch
//      exists to fix. Spelling out `double`, `float`, `unsigned int` and
//      `unsigned long long` sidesteps the question entirely.
//
// WHY THE GEOMETRY SUMS ARE double AND THE COLOUR SUMS ARE NOT
//
// Every metric downstream is a covariance, cov = S2/n - (S1/n)^2, and the answer the LOD
// decision depends on is the SMALLEST eigenvalue -- a subtraction of two nearly equal
// numbers. Work it through for the case that matters, a locally planar leaf of 50k points
// in node-local coordinates with the plane near z = 0.5:
//
//   Szz ~ 0.25n = 12500, and the variance being extracted, for a plane 1e-3 of the node
//   thick, is ~1e-6 -- so Szz is needed to an absolute accuracy of n * 1e-6 = 0.05.
//   float32 accumulation of 50k terms into a sum of 12500 carries an absolute error of
//   about sqrt(n) * eps * S = 224 * 6e-8 * 12500 = 0.17.
//
// The error is three times the signal. In double the same estimate is ~1e-11. Colour is
// the opposite case: 8-bit input feeding a filtering decision, so float32 leaves three
// orders of magnitude of margin at a quarter of the storage.
//
// Coordinates MUST be normalised to node-local [0,1] before they are accumulated. Summing
// raw world coordinates -- UTM eastings in the hundreds of thousands -- destroys exactly
// the small differences the smallest eigenvalue is made of. See kernels/simlod/clod_accum.cuh.

#pragma once

namespace clod {

// Indices into NodeAccum::g. Arrays rather than named fields because the warp reduction
// and the Analysis roll-up both want to iterate them, and fifteen named fields would be
// fifteen copies of every loop body.
enum GeomSum : unsigned int {
	kSumX = 0, kSumY, kSumZ,
	kSumXX, kSumXY, kSumXZ,
	kSumYY, kSumYZ, kSumZZ,
	kNumGeomSums
};

enum ColorSum : unsigned int {
	kSumR = 0, kSumG, kSumB,
	kSumRR, kSumGG, kSumBB,
	kNumColorSums
};

enum AccumState : unsigned int {
	kAccumOpen      = 0,  // still receiving points; sums are partial
	kAccumClosed    = 1,  // no more points due                      (Analysis writes this)
	kAccumFinalized = 2,  // roll-up done, `score` is meaningful      (Analysis writes this)
};

// One per node in the flat pool, indexed by node index. 112 B.
//
// `state` and `score` are allocated here and written by NOBODY in the accumulator layer.
// They exist so the Analysis kernel can land without changing this layout, and so an
// inner node -- whose entry the accumulator keeps at zero -- already has somewhere for a
// rolled-up result to go if that is what Analysis decides to do.
struct NodeAccum {
	double       g[kNumGeomSums];   // 72 B: Sx Sy Sz Sxx Sxy Sxz Syy Syz Szz, node-local [0,1]
	float        c[kNumColorSums];  // 24 B: Sr Sg Sb Srr Sgg Sbb, channels in [0,1]
	unsigned int count;             // n, the number of points folded in
	unsigned int lastTouchedBatch;
	unsigned int state;             // AccumState
	float        score;             // Analysis writes this; 0 until then
};

// Read back once per launch. Everything but the first two fields is verification, which
// is the point: tests/unit/ is empty, so the invariants have to be observable from the
// host or they are not checked at all.
//
//   sumLeafCounts  must equal Stats::numPoints exactly -- it checks the accumulation
//                  against the INSERTION it shadows, per point, over the whole cloud.
//   innerWithSums  must be zero -- an inner node holding sums means a split was missed
//                  by the clear pass, which would silently poison a roll-up.
//
// Both are recomputed from scratch by the stats pass every launch. mortonWatermark,
// numAccumulated and numAtomicGroups are cumulative for the whole build.
//
// (Departure from plans/03_AccumulatorHook.md §4, which specified 32 B: numAtomicGroups
// is added because §7's third check -- the warp-aggregation ratio -- has no other way to
// be measured, and §3.5 says its expected value is unknown rather than "most of a warp".)
struct AccumGlobals {
	unsigned long long mortonWatermark;  // max Morton code seen, X-major, at MAX_DEPTH
	unsigned long long numAccumulated;   // points folded in, cumulative
	unsigned long long numAtomicGroups;  // atomic groups issued; ratio vs the above
	unsigned long long sumLeafCounts;    // recomputed per launch
	unsigned int       innerWithSums;    // recomputed per launch
	unsigned int       pad0;
};

// The host sizes its allocation from these, and the device indexes with them. A drift
// between the two sides is a silent out-of-bounds device write, so it is a compile error
// on whichever side is built first instead -- the same discipline simlod_layout.h applies
// to Node.
static_assert(sizeof(NodeAccum) == 112, "NodeAccum layout changed; see plans/03_AccumulatorHook.md §4");
static_assert(sizeof(AccumGlobals) == 40, "AccumGlobals layout changed");

}  // namespace clod
