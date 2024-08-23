#include <iostream>
#include <memory>
#include <random>
#include <filesystem>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
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
  if (argc < 2)
  {
    std::cout << "Usage: " << argv[0] << "clip directory" << std::endl;
    return -1;
  }
  auto trackerComparator = std::make_unique<TrackerComparator>();
  std::string createdDir = createDirectoryWithTimestamp();
  trackerComparator->loadDataset(argv[1]);
  trackerComparator->setupComponents();
  trackerComparator->readFirstFrameAndInit();
  trackerComparator->runEvaluation();
  trackerComparator->saveResults(createdDir);



  return 0;
}
