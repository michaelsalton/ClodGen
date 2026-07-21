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
├── build/       # Build artifacts (generated, git-ignored)
├── Makefile
└── README.md
```

## Building

```sh
make          # release build -> build/clodgen
make debug    # debug build with symbols (-O0 -g)
make run      # build and run
make clean    # remove build artifacts
```

The compiler can be overridden, e.g. `make CXX=clang++`.
