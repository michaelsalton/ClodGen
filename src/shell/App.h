// The application: owns the window, the CUDA context, the point source, the
// pipeline registry, and the shared per-frame uniforms.
//
// This is what SimLOD's main_progressive_octree.cpp is instead of: 1624 lines in one
// file with roughly forty file-scope globals, whose names are hardcoded into
// updateOctree/renderCUDA/resetCUDA. Everything a pipeline needs arrives through a
// FrameContext parameter, which is what allows more than one pipeline to exist.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "clod/CudaContext.h"
#include "clod/GLInterop.h"
#include "clod/HostDeviceCommon.h"
#include "clod/ILodPipeline.h"
#include "clod/PipelineRegistry.h"
#include "clod/PointSource.h"
#include "shell/GLRenderer.h"

namespace clod {

struct AppOptions {
	std::vector<std::string> files;      // --open
	uint64_t syntheticPoints = 0;        // --synthetic N (0 = off)
	std::string pipeline = "flat";       // --pipeline
	int width = 1600;
	int height = 900;
	bool strictTiming = false;           // --strict-timing

	// --dump-frame <path.ppm>: render, write the colour attachment, exit.
	//
	// This is how the renderer gets verified without a human looking at a window,
	// and it is the seed of the golden-image comparison in the benchmark harness.
	// Neither upstream can do this at all -- SimLOD's loop() calls exit() and has no
	// path that reaches a file write.
	std::string dumpFramePath;
	int dumpAfterFrames = 8;
	// --dump-ui: capture the presented backbuffer (ImGui included) rather than just the
	// rendered cloud. For inspecting the interface; golden images want the cloud alone.
	bool dumpIncludeGui = false;  // let the camera settle and NVRTC finish first

	// --switch-to <id> --switch-after <n>: request a runtime pipeline switch after n
	// frames.
	//
	// Exists so the switch path is testable without a human clicking a radio button.
	// Written after that path shipped broken: drawGui held a raw pointer to the pipeline
	// it was destroying, and it crashed on the first click. A GUI-only code path is an
	// untested code path.
	std::string switchToPipeline;
	int switchAfterFrames = 0;
};

// Shared shading/LOD settings. These live here, NOT in the pipeline, precisely so
// that every pipeline is guaranteed the same values on the same frame -- which is
// what makes an A/B attributable to the LOD algorithm.
struct SharedSettings {
	float lodPixelBudget = 128.0f;
	float minNodeSize = 64.0f;
	float lodScale = 0.5f;
	int pointSize = 1;
	int colorMode = COLOR_RGB;
	bool doUpdateVisibility = true;
	bool useHighQualityShading = false;
	bool enableEDL = true;
	float edlStrength = 0.4f;
	bool showBoundingBox = false;
	bool showPoints = true;
};

// Scan a directory for loadable point clouds. Free function so `--list-datasets` can use
// it without constructing an App (which would open a window and a CUDA context).
struct DatasetEntry {
	std::string path;        // absolute, what gets loaded
	std::string label;       // relative to the scanned directory
	uint64_t bytes = 0;
	uint64_t numPoints = 0;  // exact for .simlod, 0 when unknown
	bool supported = false;
	std::string note;        // why unsupported, or extra detail
};
std::vector<DatasetEntry> scanDatasetDir(const std::string& dir);

class App {
public:
	App();
	~App();

	bool init(const AppOptions& options, std::string* err);
	int run();

private:
	void registerPipelines();
	bool loadCloud(const std::vector<std::string>& files, std::string* err);
	bool loadSynthetic(uint64_t numPoints, std::string* err);
	bool activateCloud(std::string* err);

	SharedUniforms buildUniforms() const;
	DeviceBudget computeBudget() const;

	void drawGui();

	// Reads back the colour attachment and writes a binary PPM. Returns false and
	// logs on failure.
	bool dumpFrame(const std::string& path);

	AppOptions m_options;
	SharedSettings m_settings;

	std::unique_ptr<CudaContext> m_cuda;
	GLRenderer m_renderer;
	GLInterop m_interop;

	PipelineRegistry m_registry;
	std::unique_ptr<PointSource> m_source;
	CloudMeta m_meta;
	DeviceBudget m_budget;

	std::string m_status;   // shown in the GUI; errors are not silently swallowed
	bool m_statusIsError = false;

	std::vector<DatasetEntry> m_datasets;
	std::string m_datasetDir = "data";
	bool m_datasetsScanned = false;
	int m_selectedDataset = -1;   // index into m_datasets, for the combo preview

	// A cloud load REQUESTED by the GUI.
	//
	// Deferred for the same reason as a pipeline switch: loading calls switchTo(), which
	// destroys the active pipeline, and drawGui holds a raw pointer to it.
	//
	// The extra frame of delay is so the "loading..." status is actually presented before
	// the blocking read starts. Ingest is synchronous, so a 5.3GB file freezes the window
	// for several seconds; without the delay the user gets no feedback at all until it is
	// already done.
	std::string m_pendingLoadPath;     // empty = none; "synthetic:<n>" for the fixture
	int m_pendingLoadDelayFrames = 0;

	void scanDatasets();
	void applyPendingLoad();
	void requestLoad(const std::string& pathOrSynthetic, const std::string& label);

	// A pipeline switch REQUESTED by the GUI, applied at the top of the next frame.
	//
	// Not applied inline, because switching destroys the active pipeline and drawGui
	// holds a raw ILodPipeline* for the whole of its body -- rendering stats and calling
	// pipeline->gui() after the switch would dereference freed memory. It also avoids
	// tearing down device buffers halfway through a frame that has already issued a
	// render against them.
	std::string m_pendingPipeline;

	void applyPendingPipelineSwitch();

	uint64_t m_frameCounter = 0;
	SharedUniforms m_frozen = {};
	bool m_hasFrozen = false;
};

}  // namespace clod
