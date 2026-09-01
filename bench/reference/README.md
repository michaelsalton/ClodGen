# Reference baselines

Upstream SimLOD / CudaLOD numbers captured on this machine, via `make simlod` and
`make cudalod`. These are the oracle ClodGen's ported pipelines are validated
against — not legacy cruft. Recapture after any submodule bump or driver change.

## Machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, 16303 MiB, compute capability 12.0 (`sm_120`), 84 SMs |
| Display | 3840x1600 (matters — see the render-buffer note below) |
| CUDA | 13.1 |
| Compiler | g++ 15.2.0, CMake 4.2.3 |
| Dataset | `data/morro_bay_35M/morro_bay_36M.las`, 36,200,706 points |

## CudaLOD — `cudalod_35M.txt`

Structural output, identical across every run and every sampling strategy:

| | |
|---|---|
| `#points` | 36,200,706 (exactly matches the LAS header) |
| `#voxels` | 12,742,112 |
| `#nodes` | 2,252 |
| `#allocated (splitting)` | 929,611,360 |
| `#allocated (voxelization)` | 1,402,630,112 |
| points/node min-avg-max | 1 - 20,864 - 49,739 |
| voxels/node min-avg-max | 1,046 - 24,646 - 56,067 |

Timings per sampling strategy. The app builds once in the ctor with strategy 0,
then rebuilds each time a strategy button is pressed:

| strategy | split | voxelize | total | MP/s | slab watermark |
|---|---|---|---|---|---|
| 0 `FIRST_COME` (ctor, cold) | 6.6 ms | 4.7 ms | 11.3 ms | 3,210 | 1.20 GB |
| 0 `FIRST_COME` (warm) | 5.1 ms | 4.1 ms | 9.3 ms | 3,912 | 1.20 GB |
| 2 `AVERAGE_SINGLECELL` | 4.9 ms | 20.1 ms | 25.0 ms | 1,448 | 3.82 GB |
| 3 `WEIGHTED_NEIGHBORHOOD` | 4.9 ms | 60.9 ms | 65.8 ms | 550 | 3.82 GB |

Strategy 1 (`RANDOM`) was not exercised in this capture.

Two things worth carrying into ClodGen's own benchmarking:

- **The strategies are not close.** `WEIGHTED_NEIGHBORHOOD` — the paper's quality
  contribution — voxelizes 13x slower than `FIRST_COME` (60.9 ms vs 4.7 ms) for
  bit-identical tree structure. Any "CudaLOD does N MP/s" claim is meaningless
  without naming the strategy. The README's previously recorded 3,332 MP/s was
  strategy 0.
- **Split cost is strategy-independent** (~5 ms) and the *cold* ctor build is
  ~25% slower than a warm rebuild. Measure warm, or measure both deliberately.

### Three upstream defects fixed to get this capture

The previously recorded baseline was produced with a silently misconfigured
build, and the render phase crashed. All three fixes are in
`patches/cudalod-linux-port.patch`:

1. **`MAX_BUFFER_SIZE` was shadowed.** `sampling_cuda/sampling_cuda.h`
   unconditionally `#define`d it to 2,147,483,647 and `SimLOD.h` included that
   *before* the `#ifndef`-guarded default, so the live pipeline ran with a
   2.147 GB slab regardless of intent. That dead include is now removed
   (`simlod_gentree_cuda::VoxelTreeGen` is only referenced from commented-out
   code).
2. **The override never worked.** `cmake -DMAX_BUFFER_SIZE=...` only sets a cache
   variable, so the documented knob was a no-op. It is now forwarded via
   `target_compile_definitions`, and renamed to `CUDALOD_MAX_BUFFER_SIZE` —
   `MAX_BUFFER_SIZE` is *also* a member of `ProgressiveFileBuffer`, so defining
   that name on the compiler command line breaks the build outright.
3. **Two unbounded-allocator overruns**, both invisible because the bump
   allocators in `lib.h.cu` have no capacity check — the symptom is a flood of
   `illegal memory access` from the render kernel, never an allocation failure:
   - The slab default is now **8 GB**, sized from the worst strategy's measured
     3.82 GB watermark rather than the default strategy's 1.20 GB. At 4 GB,
     pressing the strategy-2 button overran it.
   - `ptr_render_buffer` was a fixed 100 MB, but `renderHQS` allocates
     **28 bytes/pixel**. That is 58 MB at the author's 1920x1080 and fits; at
     3840x1600 it is 172 MB and does not. It is now sized from the actual
     framebuffer at 64 bytes/pixel.

With those, the run is clean: **0** illegal-access errors across all four builds.

## ClodGen's port vs the reference

`clodgen --pipeline cudalod --open data/morro_bay_35M/morro_bay_36M.simlod --dump-frame x.ppm`

| metric | reference | ClodGen port | |
| ------ | --------- | ------------ | --- |
| points | 36,200,706 | 36,200,706 | exact |
| nodes | 2,252 | 2,252 (517 inner, 1,735 leaves) | exact |
| max points/node | 49,739 | 49,739 | exact |
| watermark, split | 929,611,360 | 929,611,360 | exact |
| watermark, voxelize | 1,402,630,112 | 1,402,6xx,xxx | exact to 3 s.f. |
| build, strategy 0 | 9.3 – 11.3 ms | 9.66 ms | within range |
| voxels | 12,742,112 | 12,742,500 | **+388 (+0.003%)** |

The port is deterministic (identical across runs), and the device code is vendored
unmodified, so the tree really is the same tree.

**The voxel delta is an input-provenance difference, not a port defect.** The reference
reads the `.las`, whose header carries an f64 extent of `1399.9900000002235`; ClodGen
currently reads the `.simlod`, whose header is f32 (`1399.989990234375`). CudaLOD derives
`cubeSize` from the longest axis of that box, so a last-bit difference shifts the 128³
voxel grid's cell boundaries and a few hundred points fall in different cells. Confirming
this requires the `.las` reader, so that the identical file can be loaded — until then it
is an explanation, not a closed issue.

Worth remembering as a general lesson for this project: bounding-box provenance is part of
the input. Two readers of the same points can produce trees that differ slightly, which is
exactly why `CloudMeta` keeps the translation in f64 and snaps it to a power of two.

## SimLOD

Not yet captured — SimLOD accepts no command-line arguments and loads only via
drag-and-drop onto its window, so it cannot be driven from a script. Run
`make simlod`, drop `data/morro_bay_35M/morro_bay_36M.simlod` on the window, and
save the stats panel output to `simlod_35M.txt`.
