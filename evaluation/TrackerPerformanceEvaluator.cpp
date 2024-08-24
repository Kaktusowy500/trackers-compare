#include "TrackerPerformanceEvaluator.hpp"
#include <fstream>
#include <numeric>
#include <iostream>

TrackerPerformanceEvaluator::TrackerPerformanceEvaluator(const std::string &tracker_name) : tracker_name(tracker_name) {};

// Private helper method to calculate the Intersection over Union (IoU) or overlap
double TrackerPerformanceEvaluator::calculateOverlap(const cv::Rect &groundTruth, const cv::Rect &trackingResult)
{
  int intersectionArea = (groundTruth & trackingResult).area();
  int unionArea = groundTruth.area() + trackingResult.area() - intersectionArea;
  return static_cast<double>(intersectionArea) / unionArea;
}

// Private helper method to calculate the center localization error
double TrackerPerformanceEvaluator::calculateCenterError(const cv::Rect &groundTruth, const cv::Rect &trackingResult)
{
  cv::Point2f gtCenter = (groundTruth.tl() + groundTruth.br()) * 0.5;
  cv::Point2f trCenter = (trackingResult.tl() + trackingResult.br()) * 0.5;
  return cv::norm(gtCenter - trCenter);
}

// Method to add a single frame's results
void TrackerPerformanceEvaluator::addFrameResult(const cv::Rect &groundTruth, const cv::Rect &trackingResult,
                                                 double processing_time, bool valid)
{
  FrameResult result;
  result.overlap = calculateOverlap(groundTruth, trackingResult);
  result.error = calculateCenterError(groundTruth, trackingResult);
  result.processing_time = processing_time;
  result.bbox_area = trackingResult.area();
  result.valid = valid;
  results.push_back(result);
}

// Method to calculate and return the average overlap
double TrackerPerformanceEvaluator::getAverageOverlap() const
{
  double sumOverlap = 0.0;
  int validCount = 0;

  for (const auto &result : results)
  {
    if (result.valid)
    {
      sumOverlap += result.overlap;
      validCount++;
    }
  }

  return validCount > 0 ? sumOverlap / validCount : 0.0;
}

// Method to calculate and return the average center error
double TrackerPerformanceEvaluator::getAverageError() const
{
  double sumError = 0.0;
  int validCount = 0;

  for (const auto &result : results)
  {
    if (result.valid)
    {
      sumError += result.error;
      validCount++;
    }
  }

  return validCount > 0 ? sumError / validCount : 0.0;
}

double TrackerPerformanceEvaluator::getAverageProcessingTime() const
{
  double sumProcessingTime = 0.0;
  int validCount = 0;

  for (const auto &result : results)
  {
    if (result.valid)
    {
      sumProcessingTime += result.processing_time;
      validCount++;
    }
  }

  return validCount > 0 ? sumProcessingTime / validCount : 0.0;
}

// Method to save the results to a file
void TrackerPerformanceEvaluator::saveResultsToFile(const std::string &filename) const
{
  std::ofstream file(filename);
  if (!file.is_open())
  {
    std::cerr << "Could not open the file: " << filename << std::endl;
    return;
  }

  file << "Frame,Overlap,Center Error,Processing Time,BBox Area,Valid" << std::endl;
  for (size_t i = 0; i < results.size(); ++i)
  {
    file << i + 1 << "," << results[i].overlap << "," << results[i].error << "," << results[i].processing_time << "," << results[i].bbox_area << "," << results[i].valid << "\n";
  }

  std::cout << "\nTracker: " << tracker_name << " statistics: " << std::endl;
  std::cout << "Average Overlap: " << getAverageOverlap() << "\n";
  std::cout << "Average Center Error: " << getAverageError() << "\n";
  std::cout << "Average Center Error: " << getAverageProcessingTime() << "\n";

  file.close();
}
