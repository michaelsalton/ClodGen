Rasterize Kernel (SimLOD) - Every frame unconditionally
    LOD Selection - frustum test, descend while node > 128px
    Drawing - one block per visible node, atomicMin splatting

Update Kernel (SimLOD + accumulator hook) - Only if a batch completed - per point
    Expand octree - count → make room for new points
    Voxel sampling - build the coarse version
    Allocate chunks - acquire required storage
    Insert points and voxels - write everything

Analysis Kernel (Custom) - Every frame unconditionally - per node
    Advance the watermark
    Test closure
    Roll up statistics - leaf accumulators summed up the tree, then finalize
    Score and enqueue - geometric score x screen coverage, rebuilt each frame

Refinement Kernel (Custom) - Runs if budget remains, and may produce no changes
    Color filtering
    Deepening - local split and redistribute
    Collapsing - requires a grid pool first
    Compression - lags closure by a widen margin
