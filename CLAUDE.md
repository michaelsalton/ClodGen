# ClodGen

A real-time LOD generation and rendering program for lidar point clouds. It extends
SimLOD and borrows from CudaLOD. **The project's goal is to make SimLOD's octree
construction detail-aware** — geometry analysis driving per-node sample budgets and
depth, computed online while the cloud is still streaming in.

See [README.md](README.md) for the idea, the kernel architecture, build details and
current status.

## What the work actually is

Four kernels, three cadences. The first two exist; the last two are the project.

| kernel | when | state |
| --- | --- | --- |
| Rasterize (SimLOD) | every frame | exists — `kernels/simlod/simlod_render.cu` |
| Update (SimLOD + accumulator hook) | on batch completion | exists, hook does not |
| Analysis | every frame, per node | **to build** |
| Refinement | if budget remains | **to build** |

Analysis is cheap, unconditional and does not mutate the tree. Refinement is the only
thing that mutates it and the only thing that consumes budget — which is what makes
"how much refinement budget the system got" the single independent variable of the
evaluation. Keep that separation.

Read `plans/tempnotes.md` for the current shape of the design, and
`plans/01_NoveltyAssessment.md` before touching LOD selection metrics, node budgets or
split criteria.

## The shared path

**Everything outside LOD generation stays shared.** One loader, one camera, one
rasteriser (`kernels/shared/`), one pixel budget and one device-memory budget handed
unchanged to every pipeline. A change that gives one pipeline its own render path, its
own budget interpretation or its own loader destroys the only baseline the research has.

If a pipeline genuinely needs different behaviour, that is a finding to surface, not a
special case to add. Say so rather than diverging the shared path.

Corollary: `flat` (no LOD) is the control condition and the image-quality ground truth.
It is not dead code and is not a candidate for optimisation-by-deletion.

## Working here

```sh
make                       # release -> build/clodgen
make debug                 # -O0 -g -> build-debug/
make run ARGS="--open data/morro_bay_35M/morro_bay_36M.simlod"
./build/clodgen --check-kernels    # compile every kernel headlessly; exits non-zero on failure
./build/clodgen --dump-frame ...   # headless frame capture for verification
```

**A successful `make` does not mean the kernels compile.** Every `.cu` under `kernels/`
is compiled at runtime by NVRTC — that is what makes hot reload work, and it means
kernel errors are invisible to the build. After touching anything under `kernels/`, run
`--check-kernels`. It needs no display, GPU context or point cloud.

There is no test suite yet: `tests/unit/` is empty and `CLODGEN_BUILD_TESTS` is OFF, so
`make test` currently runs ctest against nothing. Verification today is
`--check-kernels`, `--dump-frame` and the structural counts in `bench/reference/`.

Only `.simlod` loads — the `.las` / `.laz` readers are not implemented. Data lives under
`data/` and is gitignored; do not assume a cloud is present.

## Layout

| Path | Contents |
| --- | --- |
| `include/clod/` | Public headers — the pipeline SDK (`ILodPipeline.h` is the contract) |
| `src/shell/` | Window, orbit camera, ImGui panel, pipeline registry |
| `src/cuda/` | NVRTC wrapper, CUDA context, GL interop |
| `src/io/` | Point cloud readers |
| `src/pipelines/` | Host side of each pipeline (flat, cudalod, simlod) |
| `kernels/shared/` | Rasteriser + allocators every pipeline includes — **shared, keep it that way** |
| `kernels/{flat,cudalod,simlod}/` | Device code per pipeline |
| `bench/reference/` | Upstream baseline numbers the ports are validated against |
| `plans/` | Research direction and staged implementation plans |
| `references/` | Source papers |
| `external/`, `patches/` | Submodules and the Linux/CUDA-13 fixes applied to them |

## Rules for vendored code

- Copied files carry a three-line header naming upstream path, commit and copyright.
  Keep it when moving or editing them, and add one when vendoring something new.
- Vendored device code is kept **byte-identical** to upstream where possible, so it
  stays verifiable against `bench/reference/`. Prefer adding a separate pass over
  editing an upstream kernel.
- **Licence trap:** one upstream file is CC BY-NC-SA and is deliberately not used,
  because it would infect this project's licence. Check
  [THIRD_PARTY.md](THIRD_PARTY.md) before copying anything new out of `external/`.
- Kernels must be taken from the **patched** submodule tree — the Linux port fixes a
  `typedef char int8_t` that otherwise collides with `cuda/std/cstdint`.

## Constraints the detail-aware work runs into

These are load-bearing and were established by reading the code; check them before
proposing a design that assumes otherwise.

- **Per-node accumulators do not belong on `Node`.** SimLOD's `Node` is 152 bytes,
  asserted in `kernels/simlod/structures.cuh` and mirrored host-side in
  `kernels/simlod/simlod_layout.h` (`kNodeBytes`) to size the 200k-node pool. Widening
  it touches both and changes a vendored struct. Use a **side array indexed by node
  index**, allocated by the host and passed to the new kernels.
- **The accumulator hook site is inside vendored code.** The natural place to accumulate
  is `sampleVoxel` / `insertPoints` in `kernels/simlod/progressive_octree_voxels.cu`,
  which is kept byte-identical. Two acceptable routes: a separate pass that re-traverses
  the batch, or a `#ifdef`-guarded macro so the default build emits identical device code
  and the instrumented build is a declared separate mode (the pattern
  `plans/02_ProfilingTools.md` Layer 3 already establishes).
- **Colour averaging has a memory wall.** The occupancy grid is 1 bit per cell (256 KB
  per inner node at 128³). An RGBA-sum grid at the same resolution would be ~16 MB
  *per node*. Filtering therefore needs a sparse accumulator keyed off the voxel
  backlog, not a dense per-cell one — or it pays CudaLOD's cost (3.2× device memory,
  ~13× voxelisation time).
- **Collapsing needs a grid pool first.** Point/voxel chunks are already recycled
  through `chunkQueue` when a leaf splits, but the 256 KB `OccupancyGrid` allocated on
  split is never freed. There is nothing to return it to.
- **The node pool has no device-side capacity check.** `numNodes` is a bump index grown
  by `atomicAdd(&stats->numNodes, 8)`; the host clamping in `SimlodPipeline::readStats`
  is the only place exhaustion is noticed. Anything that splits more eagerly must keep
  that reporting intact.
- **Uniform control flow around `ClodAllocator`.** It is deliberately non-atomic: every
  thread walks the identical allocation sequence. No `alloc()` behind a branch, in a
  data-dependent condition, or in a loop with a varying trip count. See the banner in
  `kernels/shared/clod_alloc.cuh`.

## Reporting numbers

- **Always name CudaLOD's sampling strategy when quoting a throughput figure.** The tree
  is bit-identical across all four, but `WEIGHTED_NEIGHBORHOOD` voxelizes ~13× slower
  than `FIRST_COME` and needs 3.2× the device memory. An unqualified "CudaLOD does X
  MP/s" is meaningless. The same will be true of any SimLOD number without the batch
  size and the device-side time budget.
- **The current pipeline comparison is not a clean quality A/B.** Both selection passes
  take the shared pixel budget, but SimLOD's projects all eight corners and takes the
  screen AABB while CudaLOD's estimates from the node centre, so they do not interpret
  it identically. Do not present those numbers as a quality result.
- The native metrics (`CLOD_LOD_SIMLOD_NATIVE`, `CLOD_LOD_CUDALOD_NATIVE`) exist in the
  kernels for validating a port against its published behaviour, but no pipeline
  populates `KernelProgramDesc::defines`, so that path is currently unreachable from the
  host.
- Recapture `bench/reference/` after a submodule bump or driver change.

## Known traps

- **CudaLOD faults on `--synthetic`** with `CUDA_ERROR_ILLEGAL_ADDRESS` in the split
  kernel, at every point count tried. Not root-caused; coplanarity and bbox-boundary
  causes ruled out. The pipeline is refused for that fixture on purpose. Use a real cloud.
- A device fault is unrecoverable, so ClodGen exits immediately naming the kernel rather
  than continuing. Continuing previously produced a cascade of errors, then host heap
  corruption, then a SIGSEGV in a file-watcher thread — a trail pointing nowhere near the
  cause. Keep that fail-fast behaviour.
- Timing instrumentation is incomplete: only `FlatPipeline` fills `renderDeviceMsLast`,
  so the GUI prints a plausible-looking `0.00` for the other two. Treat render-time
  numbers from `simlod`/`cudalod` as absent, not zero. See `plans/02_ProfilingTools.md`.
- A GUI-only code path is an untested code path. `--switch-to` / `--switch-after` and
  `--dump-frame` exist so the switch and render paths are scriptable; keep new controls
  reachable from the command line.

## Deeper context — read when relevant

- `plans/01_NoveltyAssessment.md` — the research thesis: content-adaptive per-node point
  budgets computed online during streaming LOD construction, the prior art it must be
  differentiated from (Sequential Point Trees, VoxelMap, Pauly et al.), and the
  evaluation plan. **Read before any work on LOD selection metrics, node budgets, or
  split criteria.**
- `plans/02_ProfilingTools.md` — staged plan for the measurement layer. **Read before
  touching timing, benchmarking or `ILodPipeline`'s counters.**
- `bench/reference/README.md` — capture methodology and the machine baselines were taken on.
- `references/` — source papers (SimLOD, CudaLOD).
