# ClodGen

A point cloud viewer with **swappable LOD generation pipelines**.

Loads `.las` / `.laz` / `.simlod`, renders it with a CUDA software rasteriser, and
lets you toggle at runtime between LOD generation algorithms — SimLOD's progressive
octree, CudaLOD's batch counting-sort + voxelisation, and your own — on the same
loader, the same camera and the same rasteriser, so a difference in the output is
attributable to the LOD algorithm rather than to incidental plumbing.

That last part is the whole point. The two reference implementations are separate
binaries with different loaders, different LOD metrics (`dx > 2 * minNodeSize` in
world units versus `cubeSize / distance < 1 - 0.97 * LOD`, angular) and different
rasterisers (a packed `uint64` `atomicMin` framebuffer with a divide-by-count
resolve, versus a `uint32` depth buffer plus a 16 byte/pixel accumulator with a
Gaussian 3×3 splat). Feed those two the same octree and the images differ for
reasons that have nothing to do with LOD quality.

## Status

Working today: **three pipelines, switchable at runtime** — `flat` (no LOD, the control),
`cudalod` (batch), `simlod` (progressive). On `morro_bay_36M.simlod` at a 128 px LOD budget,
same camera, same rasteriser:

| pipeline | tree nodes | nodes drawn | samples drawn | coverage | build |
| -------- | ---------- | ----------- | ------------- | -------- | ----- |
| flat | – | 553 | 36,200,706 | 16.6 % | – |
| cudalod | 2,252 | 10 | 156,496 | 13.1 % | 9.4 ms |
| simlod | 4,137 | 68 | 1,750,475 | 16.7 % | 43.2 ms (4 launches) |

SimLOD reaches flat's coverage with 20x fewer samples; CudaLOD is far more aggressive at
231x fewer. **Do not read that as a clean quality A/B yet** — the two selection rules
measure "projected size" differently (CudaLOD estimates from the node centre, SimLOD
projects all eight corners and takes the screen-space AABB), so they do not interpret the
shared pixel budget identically. Closing that gap is the next step for making the
comparison honest.

Also working:

- Window, orbit camera, ImGui panel, dataset dropdown, `--open` / drag-and-drop
- `.simlod` loading; 36,200,706 points from `morro_bay_36M.simlod` verified exactly
- CUDA software rasteriser: packed `uint64` `atomicMin` framebuffer, EDL,
  `surf2Dwrite` into a GL texture
- The `flat` pipeline (no LOD) as the control condition and image-quality ground truth
- Kernel hot reload: edit a `.cu` or `.cuh`, save, and the next frame runs it — a
  broken kernel is **non-fatal** and leaves the previous version live
- `--dump-frame` for headless verification

Next: `.las` / `.laz` readers; making the two LOD metrics genuinely comparable; the real
streaming loader (SimLOD currently builds progressively from an already-resident cloud, which
measures construction but not the paper's loading/generation overlap).

## Requirements

- A C++23 compiler (`g++` 13+; the vendored `unsuck.hpp` uses `<print>`/`<format>`)
- CMake ≥ 3.22, GNU Make
- A CUDA toolkit ≥ 12.4 (driver API, NVRTC, nvJitLink). No `.cu` is compiled at
  build time — every kernel is compiled at runtime by NVRTC, which is what makes
  hot reload possible.
- GL development headers **including GLU**:
  ```sh
  sudo apt install cmake cuda-toolkit-13-1 libglu1-mesa-dev
  ```
  `libglu1-mesa-dev` is separate from `libgl1-mesa-dev` and easy to miss — the
  vendored `glew.h` includes `GL/glu.h` unconditionally.

## Building and running

```sh
make                  # release build -> build/clodgen
make debug            # -O0 -g into build-debug/
make run ARGS="--open data/morro_bay_35M/morro_bay_36M.simlod"
make clean
```

`make` is a thin façade over CMake; the `simlod` / `cudalod` targets below are
unchanged. `clodgen --help` lists the options.

## Layout

```
.
├── include/clod/    Public headers = the pipeline SDK
├── src/
│   ├── shell/       Window, camera, GUI, pipeline registry
│   ├── cuda/        NVRTC wrapper, CUDA context, GL interop
│   ├── io/          Point cloud readers
│   └── pipelines/   Host side of each LOD pipeline
├── kernels/         Device code, NVRTC-compiled at runtime and hot-reloaded
│   ├── shared/      The rasteriser and allocators every pipeline #includes
│   ├── flat/        The no-LOD control pipeline
│   ├── cudalod/     Batch pipeline (vendored kernels + our selection pass)
│   └── simlod/      Progressive pipeline (vendored kernels + our selection pass)
├── bench/reference/ Upstream baselines this is validated against
├── external/        SimLOD, CudaLOD, glfw (git submodules)
└── patches/         Linux/CUDA-13 fixes applied to the submodules at build time
```

## Provenance

ClodGen vendors code from both reference implementations, which are MIT. Every
copied file carries a three-line header naming its upstream path, commit and
copyright. See [THIRD_PARTY.md](THIRD_PARTY.md) for the full record — including one
file that is deliberately **not** used, because it is CC BY-NC-SA and would infect
this project's licence.

## Submodules

| Path                 | Upstream                                  | Branch/Tag | Notes                          |
| -------------------- | ----------------------------------------- | ---------- | ------------------------------ |
| `external/SimLOD`    | https://github.com/m-schuetz/SimLOD       | `ubuntu`   | CUDA point-cloud LOD renderer  |
| `external/CudaLOD`   | https://github.com/m-schuetz/CudaLOD      | `main`     | Earlier CUDA LOD work + paper  |
| `external/glfw`      | https://github.com/glfw/glfw              | `3.4`      | Windowing; see THIRD_PARTY.md  |

`clodgen` references `external/*/libs/**` only, and never a file that
`patches/*.patch` modifies — so it builds after a bare
`git submodule update --init`, regardless of whether the submodule patches have
been applied.

Clone with submodules:

```sh
git clone --recurse-submodules git@github.com:michaelsalton/ClodGen.git
```

Already cloned without them:

```sh
git submodule update --init --recursive
```

Update one to the latest upstream commit on its tracked branch:

```sh
git submodule update --remote external/SimLOD
git add external/SimLOD && git commit -m "Bump SimLOD"
```

## Running the subrepos

```sh
make subrepos   # list what's wired up
make simlod     # patch, configure, build and run SimLOD
make cudalod    # explains why there is no Linux build
```

### SimLOD

Requires `cmake`, a CUDA toolkit (>= 12.4, for `nvrtc` + `nvJitLink`), and GL
development headers including GLU:

```sh
sudo apt install cmake cuda-toolkit-13-1 libglu1-mesa-dev
```

`libglu1-mesa-dev` is separate from `libgl1-mesa-dev` and is easy to miss —
the bundled `glew.h` includes `GL/glu.h` unconditionally.

`make simlod` then configures, builds, and launches it. It opens an empty
window — **drag a `.las` / `.laz` / `.simlod` file onto it** to load a point
cloud. SimLOD accepts no command-line arguments.

Overridable variables:

| Variable          | Default        | Purpose                                     |
| ----------------- | -------------- | ------------------------------------------- |
| `CUDA_PATH`       | `/usr/local/cuda` | Toolkit location; SimLOD also reads this at runtime |
| `SIMLOD_GPU_ARCH` | `compute_120`  | NVRTC target arch (`compute_120` = Blackwell / RTX 50xx) |

Upstream's `ubuntu` branch does not build as-is, so
[patches/simlod-linux-sm120.patch](patches/simlod-linux-sm120.patch) is applied
automatically (idempotently) by `make simlod`. It makes three changes:

1. Links `CUDA::nvJitLink` — used by `CudaModularProgram.h` but never linked.
2. Falls back to `/usr/local/cuda` when `CUDA_PATH` is unset, instead of
   constructing a `std::string` from a null pointer.
3. Reads the NVRTC target arch from `SIMLOD_GPU_ARCH` instead of the
   hardcoded `compute_89`, which does not match Blackwell GPUs.
4. Calls the CUDA 13 `cuCtxCreate` (`_v4`), which takes an extra
   `CUctxCreateParams*`. Guarded on `CUDA_VERSION` so CUDA 12 still builds.
5. Adds `-I $CUDA_PATH/include/cccl` to the NVRTC options. CUDA 13 relocated
   the libcu++ headers (`<cuda/std/*>`, pulled in by `cooperative_groups`)
   into `include/cccl`, so runtime kernel compilation fails without it.

To rebuild from scratch: `make simlod-clean && make simlod`.

### CudaLOD

Upstream is Visual Studio only — it ships `build/CudaLOD.sln` and `.vcxproj`
with no CMake or Makefile. [patches/cudalod-linux-port.patch](patches/cudalod-linux-port.patch)
adds a Linux port, applied automatically (idempotently) by `make cudalod`.

Same dependencies as SimLOD. CudaLOD takes no command-line arguments and its
input path is hardcoded upstream, so the patch adds a `CUDALOD_LAS` override.
`make cudalod` defaults to the smallest `.las` under `data/`, so with a point
cloud there it just runs:

```sh
make cudalod                                    # smallest cloud in data/
make cudalod CUDALOD_LAS=/path/to/cloud.las     # explicit
```

Note the build directory is `cmake-build/`, not `build/` — upstream tracks
`build/` for the solution files.

| Variable             | Default        | Purpose                                  |
| -------------------- | -------------- | ---------------------------------------- |
| `CUDALOD_LAS`        | smallest `data/**/*.las` | Input `.las` point cloud        |
| `CUDALOD_GPU_ARCH`   | `compute_120`  | NVRTC target arch                        |
| `CUDALOD_MAX_BUFFER` | `8'000'000'000` | Device slab bytes; must be ≤ ~2/3 of VRAM. Size it for the *worst* sampling strategy, not the default one — see [bench/reference/README.md](bench/reference/README.md) |

What the patch changes:

1. **Adds `CMakeLists.txt`**, mirroring the `ClCompile` list from
   `CudaLOD.vcxproj`. GLFW is fetched from source (the vendored lib is a
   prebuilt `msvc2017_x64` binary) and `libs/laszip` is built via
   `add_subdirectory`.
2. **Adds `cmake/openvr_stub.cpp`.** Only a Windows `openvr_api.lib` is
   vendored, so the 10 OpenVR entry points are stubbed to report "no HMD". VR
   is gated behind `Renderer::vrEnabled` (off by default), so those code paths
   never execute — the stubs exist purely to satisfy the linker.
3. **Windows path separators** in 10 headers: `#include "GL\glew.h"` → `GL/glew.h`.
4. **`min(int64_t, 1'000'000'000ll)`** in four places. `int64_t` is `long` on
   LP64, so an `ll` literal breaks template deduction.
5. **CUDA 13 API updates**: `cuCtxCreate` `_v4`, and NVVM → PTX for runtime
   linking. `nvrtcGetNVVM` was removed and there is no `CU_JIT_INPUT_LTOIR`
   for `cuLinkAddData`, so `-dlto` is dropped and PTX is linked instead.
6. **`-I $CUDA_PATH/include/cccl`** for both host (thrust) and device
   (cooperative_groups) compilation, plus `typedef char int8_t` →
   `signed char` in two kernel headers, which otherwise conflicts with
   `cuda/std/cstdint`.
7. **`toClipboard`** implemented for Linux via `glfwSetClipboardString`; the
   platform layer only had a Win32 version.
8. **`MAX_BUFFER_SIZE`** made overridable, defaulting to 10GB rather than
   upstream's 15GB (which assumes a 24GB+ card).
9. **`CUDALOD_LAS`** env override plus an existence check, replacing the
   hardcoded `D:/dev/pointclouds/...` paths.
10. **Symlinks** `modules/` and `shaders/` into the build dir instead of
    copying. The `.cu` files are compiled at runtime and are not build inputs,
    so a `POST_BUILD` copy goes stale whenever a kernel is edited.

11. **Three latent-bug fixes** needed to get a clean, honest run: the shadowed
    `MAX_BUFFER_SIZE`, an override that never reached the compiler, and two
    unbounded-allocator overruns (the device slab, and a fixed 100MB render
    buffer against a renderer that wants 28 bytes/pixel). Each is explained at
    the point of change in the patch, and summarised in
    [bench/reference/README.md](bench/reference/README.md).

#### Verified output

On an RTX 5080 (16GB, `sm_120`) with `data/morro_bay_35M/morro_bay_36M.las`,
36,200,706 points → 12,742,112 voxels in 2,252 nodes, with **0** CUDA errors
across all four sampling strategies:

| strategy | split | voxelize | total | MP/s | slab watermark |
| -------- | ----- | -------- | ----- | ---- | -------------- |
| 0 `FIRST_COME` (cold) | 6.6 ms | 4.7 ms | 11.3 ms | 3,210 | 1.20 GB |
| 0 `FIRST_COME` (warm) | 5.1 ms | 4.1 ms | 9.3 ms | 3,912 | 1.20 GB |
| 2 `AVERAGE_SINGLECELL` | 4.9 ms | 20.1 ms | 25.0 ms | 1,448 | 3.82 GB |
| 3 `WEIGHTED_NEIGHBORHOOD` | 4.9 ms | 60.9 ms | 65.8 ms | 550 | 3.82 GB |

`#points` matches the loaded count exactly, and strategy 0's throughput is
consistent with the paper's RTX 3090 figures.

**Always name the sampling strategy when quoting a throughput number.** The tree
structure is bit-identical across all four, but `WEIGHTED_NEIGHBORHOOD` — the
paper's quality contribution — voxelizes 13× slower than `FIRST_COME`, and needs
3.2× the device memory. Sizing the slab for the default strategy is what made
the strategy buttons overrun it.

Full capture and methodology: [bench/reference/README.md](bench/reference/README.md).

## Known issues

**CudaLOD faults on the synthetic fixture.** `--synthetic` builds a cube shell plus a
helix, and CudaLOD's split kernel hits `CUDA_ERROR_ILLEGAL_ADDRESS` on that distribution
at every point count tried (200k – 36M), while real scans of the same size build fine and
match the reference exactly. The fault moves from `kernel2` to `kernel3` as the count
grows, which suggests one of upstream's unchecked device-side capacities rather than an
off-by-one — its split runs once at a fixed depth and its own comments concede it cannot
subdivide further. Ruled out so far: exact coplanarity (jitter did not help) and points
sitting on the box boundary (padding the bbox did not help). **Not root-caused.**

The pipeline is therefore refused for that fixture, with the reason in the tooltip, rather
than being allowed to take the process down. Use a real cloud with CudaLOD.

A device fault is unrecoverable — the CUDA context dies and every later call fails — so
ClodGen exits immediately with a diagnostic naming the kernel, instead of continuing. That
matters: continuing produced a cascade of identical errors, then host heap corruption, then
a SIGSEGV in a file-watcher thread, which is a trail that points nowhere near the cause.

## Checking kernels

Every `.cu` under `kernels/` is compiled at runtime by NVRTC, so a broken kernel is not
a build error. `--check-kernels` compiles and links them headlessly and exits non-zero
on failure — no window, no display, no point cloud needed:

```sh
./build/clodgen --check-kernels                       # everything under kernels/
./build/clodgen --check-kernels path/to/kernel.cu     # one file, from anywhere
./build/clodgen --check-kernels --as-group a.cu b.cu  # link them as one program
./build/clodgen --check-kernels --ptx                 # PTX + driver JIT instead of LTOIR
```

### Compile-compatibility of the reference kernels

Run before starting the pipeline ports, since it decides whether their device code can
be used as-is. Result on this machine (CUDA 13.1, `sm_120`), against the **patched**
submodule trees:

| kernels | LTOIR + nvJitLink | PTX + driver JIT |
| ------- | ----------------- | ---------------- |
| SimLOD `progressive_octree_voxels.cu`, `reset.cu`, `render.cu` | ok | — |
| CudaLOD `lib.cu` + `kernel.cu` | ok | ok |
| CudaLOD `lib.cu` + `render.cu` | ok | — |

Both survive `-default-device`, `--std=c++20` and `-dlto` unmodified, so the ports can
keep their device code byte-identical and stay verifiable against
`bench/reference/`. `LinkMode::Ptx` exists as an escape hatch but is not needed.

Two things this surfaced that matter for the ports:

- **CudaLOD's `kernel.cu` and `render.cu` are not standalone translation units.** They
  reference symbols defined in `lib.cu` and must be linked with it — hence `--as-group`.
  Checked alone they fail with unresolved externs (`vec3::vec3`, `isFirstThread`), which
  says nothing about flag compatibility.
- The kernels must be copied from the **patched** tree: the Linux port fixes
  `typedef char int8_t` → `signed char` in two of CudaLOD's kernel headers, which
  otherwise collides with `cuda/std/cstdint`.

## Baselines

`bench/reference/` holds the upstream numbers ClodGen's ported pipelines are
validated against, plus the machine they were captured on. Recapture after a
submodule bump or a driver change. See
[bench/reference/README.md](bench/reference/README.md).
