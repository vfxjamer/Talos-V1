#include "MetricSender.h"

#include "Timer.h"

#include <filesystem>

namespace py = pybind11;
using namespace GGL;

GGL::MetricSender::MetricSender(std::string _projectName, std::string _groupName, std::string _runName, std::string runID) :
	projectName(_projectName), groupName(_groupName), runName(_runName) {

	RG_LOG("Initializing MetricSender...");

	try {
		// Add common search paths to sys.path so python_scripts can be found
		auto sys = py::module::import("sys");
		auto sysPath = sys.attr("path");
		namespace fs = std::filesystem;
		std::vector<std::string> searchPaths = {
			fs::current_path().string(),
			(fs::current_path() / "python_scripts").string(),
			(fs::current_path().parent_path() / "python_scripts").string(),
			"/kaggle/working/talos",
			"/kaggle/working/talos/python_scripts",
			"/kaggle/working/talos/build",
		};
		for (const auto& p : searchPaths) {
			if (fs::exists(p)) {
				sysPath.attr("append")(p);
			}
		}
		pyMod = py::module::import("python_scripts.metric_receiver");
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to import metrics receiver, exception: " << e.what());
	}

	try {
		auto returedRunID = pyMod.attr("init")(PY_EXEC_PATH, projectName, groupName, runName, runID);
		curRunID = returedRunID.cast<std::string>();
		RG_LOG(" > " << (runID.empty() ? "Starting" : "Continuing") << " run with ID : \"" << curRunID << "\"...");

	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to initialize in Python, exception: " << e.what());
	}

	RG_LOG(" > MetricSender initalized.");
}

void GGL::MetricSender::Send(const Report& report) {
	py::dict reportDict = {};

	for (auto& pair : report.data)
		reportDict[pair.first.c_str()] = pair.second;

	try {
		pyMod.attr("add_metrics")(reportDict);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to add metrics, exception: " << e.what());
	}
}

GGL::MetricSender::~MetricSender() {
	
}