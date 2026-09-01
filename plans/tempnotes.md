ClodGen is a real-time LOD generation and rendering program made for point cloud data sets, specifically lidar data. It’s a research project that plans to extend SimLOD and draw inspiration from other papers like CudaLOD as well. 

ClodGen’s primary goal is to make the Octree construction from SimLOD into something that is detail aware using geometry analysis with metrics like Surface variation, plane residual, dimensionality features, normals, points per voxel, and color variance.

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

Covariance based
Surface variation
Plane residual
Dimensionality features
Normal
Points per voxel
Color variance

Offline quality LOD without giving up instant display
Color filtering: mentioned in SimLOD
Adaptive depth: detail where geometry warrants it, coarse where it doesn't
Potential for out-of-core rendering

Independent variable: how much refinement budget the system got
Offline testing: Build the tree offline with knowledge of the data set, apply same criterion with no closure gate and no budget - what I’m trying to approach in the real-time version
Image quality: Ground truth is a full-resolution render of the scene with no LODs
Color filtering: Goal is scene no overlapping scans, where as with first come sampling has a bias toward scans that arrived first
process the same file twice with the point order reversed. Baseline gives two visibly different colourings; filtered should give the same one


