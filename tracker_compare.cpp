#include <iostream>
#include <memory>
#include <random>
#include <filesystem>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <spdlog/spdlog.h>
#include "spdlog/cfg/env.h"
#include <yaml-cpp/yaml.h>
#include "DatasetUtils.hpp"
#include "VideoFileReader.hpp"
#include "ImageSequenceReader.hpp"

#include "TrackerComparator.hpp"

std::string createDirectoryWithTimestamp(const std::string& baseDirectory = "runs")
{
  auto now = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);

  std::tm* now_tm = std::localtime(&now_time_t);
  std::stringstream ss;
  ss << std::put_time(now_tm, "%Y-%m-%d_%H-%M-%S");

  std::string directoryName = baseDirectory + "/" + ss.str();

  std::filesystem::create_directories(directoryName);

  return directoryName;
}

int main(int argc, char** argv)
{
  spdlog::cfg::load_env_levels();
  spdlog::info("Tracker Compare started");
  YAML::Node config = YAML::LoadFile("config/general.yaml");
  
  bool preview_only = false;
  if (argc < 2)
  {
    spdlog::error("Usage: {} clip directory [-t] [tracker_for_preview_name]", argv[0]);
    return -1;
  }
  if (argc > 2 && std::string(argv[2]) == "-t")
  {
    if (argc < 4)
    {
      spdlog::error("Tracker for preview name not provided");
      return -1;
    }
    preview_only = true;
  }
  
  auto trackerComparator = std::make_unique<TrackerComparator>(config);
  trackerComparator->loadDataset(argv[1], preview_only);
  trackerComparator->setupComponents();

  if (preview_only)
  {
    trackerComparator->runPreview(argv[3]);
  }
  else
  {
    std::string createdDir = createDirectoryWithTimestamp();
    trackerComparator->runEvaluation();
    trackerComparator->saveResults(createdDir);
  }

  return 0;
}
