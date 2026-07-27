# ClodGen

A C++ research project.

## Requirements

- A C++17-capable compiler (`g++` or `clang++`)
- GNU Make

## Layout

```
.
├── src/         # Source files (.cpp)
├── include/     # Header files (.h/.hpp)
├── external/    # Third-party repos, as git submodules
├── build/       # Build artifacts (generated, git-ignored)
├── Makefile
└── README.md
```

## Submodules

Third-party sources live in `external/` as git submodules. They are *not*
compiled by this Makefile — it only globs `src/*.cpp`.

| Path                 | Upstream                                  | Notes                            |
| -------------------- | ----------------------------------------- | -------------------------------- |
| `external/SimLOD`    | https://github.com/m-schuetz/SimLOD       | CUDA point-cloud LOD renderer    |

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

## Building

```sh
make          # release build -> build/clodgen
make debug    # debug build with symbols (-O0 -g)
make run      # build and run
make clean    # remove build artifacts
```

The compiler can be overridden, e.g. `make CXX=clang++`.
