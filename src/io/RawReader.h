// Reader for SimLOD's `.simlod` format.
//
//   [24 byte header]  6 x float32 LE: min_x,min_y,min_z, max_x,max_y,max_z
//   [points]          16 bytes each: float32 x,y,z + uint8 r,g,b,a
//
// Kept because 5.6GB of it is already sitting in data/ and because it is the only
// sane benchmark input: zero parsing, ~580 MP/s, versus ~30 MP/s for .laz. Any
// throughput measurement taken through a slower reader is measuring the reader.
//
// It is an IMPORTER, not a format to build on. Three defects rule it out as our own
// cache format, and a `.clod` replacement should fix all three:
//   - the bbox is float32, so it cannot round-trip real geo coordinates;
//   - the applied translation is not recorded anywhere, so original coordinates are
//     unrecoverable;
//   - the point count is INFERRED from file size ((size - 24) / 16), so a truncated
//     file is indistinguishable from a shorter one.

#pragma once

#include <string>
#include <vector>

#include "clod/PointSource.h"

namespace clod {

// Fills meta.numPoints and meta.box*Orig, and reads all points. Coordinates are
// returned as stored -- the caller applies the translation.
bool readSimlod(const std::string& path, CloudMeta& meta,
                std::vector<Point>& points, std::string* err);

}  // namespace clod
