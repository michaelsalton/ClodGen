#include "clod/PipelineRegistry.h"

#include <cstdio>
#include <string>

#include "clod/PointSource.h"

namespace clod {

void PipelineRegistry::add(PipelineFactory factory) {
	if (!factory) return;

	std::unique_ptr<ILodPipeline> probe = factory();
	if (!probe) {
		fprintf(stderr, "clodgen: pipeline factory returned null at registration\n");
		return;
	}
	PipelineInfo info = probe->info();
	probe.reset();

	if (info.id.empty()) {
		fprintf(stderr, "clodgen: pipeline has an empty id; not registered\n");
		return;
	}
	for (const Entry& e : m_entries) {
		if (e.info.id == info.id) {
			fprintf(stderr, "clodgen: duplicate pipeline id '%s'; not registered\n",
			        info.id.c_str());
			return;
		}
	}

	m_infos.push_back(info);
	m_entries.push_back({std::move(info), std::move(factory)});
}

bool PipelineRegistry::fits(const PipelineInfo& info, const CloudMeta& meta,
                            const DeviceBudget& budget) const {
	if (info.bytesPerPointEstimate <= 0.0) return true;  // unknown, allow
	const double needed =
		info.bytesPerPointEstimate * static_cast<double>(meta.numPoints);
	return needed <= static_cast<double>(budget.bytes);
}

std::string PipelineRegistry::unsupportedReason(const PipelineInfo& info,
                                                const CloudMeta& meta,
                                                const DeviceBudget& budget) const {
	if (!fits(info, meta, budget)) {
		const double needGB =
			info.bytesPerPointEstimate * static_cast<double>(meta.numPoints) / 1e9;
		char buf[256];
		snprintf(buf, sizeof(buf),
		         "needs about %.1f GB for %llu points; the budget is %.1f GB",
		         needGB, static_cast<unsigned long long>(meta.numPoints),
		         static_cast<double>(budget.bytes) / 1e9);
		return buf;
	}

	// KNOWN INCOMPATIBILITY, kept explicit rather than discovered by crashing.
	//
	// CudaLOD's build kernels fault with CUDA_ERROR_ILLEGAL_ADDRESS on the synthetic
	// fixture, at every point count tried (200k .. 36M), while real scans of the same
	// size load and match the reference exactly. The fault moves between kernel2 and
	// kernel3 as the count grows, which points at one of upstream's several unchecked
	// device-side capacities rather than a single off-by-one; its split runs once at a
	// fixed depth and concedes in its own comments that it cannot subdivide further.
	//
	// Not yet root-caused. Refused here because a device fault is unrecoverable: without
	// this, choosing this pipeline from the dropdown terminates the process.
	if (meta.isSyntheticFixture && info.id == "cudalod") {
		return "CudaLOD's split kernel faults on the synthetic fixture's point "
		       "distribution (not yet root-caused). Load a real .simlod/.las cloud "
		       "to use this pipeline.";
	}

	return {};
}

bool PipelineRegistry::switchTo(const std::string& id, PointSource* source,
                                const CloudMeta& meta,
                                const DeviceBudget& budget, std::string* err) {
	const Entry* entry = nullptr;
	for (const Entry& e : m_entries) {
		if (e.info.id == id) {
			entry = &e;
			break;
		}
	}
	if (!entry) {
		if (err) *err = "unknown pipeline: " + id;
		return false;
	}

	if (const std::string reason = unsupportedReason(entry->info, meta, budget);
	    !reason.empty()) {
		if (err) *err = entry->info.displayName + ": " + reason;
		return false;
	}

	// Release the outgoing pipeline FIRST, so the incoming one sees the full budget.
	if (m_active) {
		m_active->release();
		m_active.reset();
		m_activeId.clear();
	}

	std::unique_ptr<ILodPipeline> pipeline = entry->factory();
	if (!pipeline) {
		if (err) *err = "factory returned null for " + id;
		return false;
	}

	if (!pipeline->initPrograms(err)) return false;
	if (!pipeline->allocate(meta, budget, err)) return false;
	pipeline->reset();

	if (source) source->rewind();

	m_active = std::move(pipeline);
	m_activeId = id;
	return true;
}

bool PipelineRegistry::reloadForCloud(PointSource* source, const CloudMeta& meta,
                                      const DeviceBudget& budget,
                                      std::string* err) {
	if (!m_active) {
		if (err) *err = "no active pipeline";
		return false;
	}
	m_active->release();
	if (!m_active->allocate(meta, budget, err)) return false;
	m_active->reset();
	if (source) source->rewind();
	return true;
}

}  // namespace clod
