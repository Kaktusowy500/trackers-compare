#include "TrackerPerformanceEvaluator.hpp"
#include <fstream>
#include <numeric>
#include <iostream>

// Private helper method to calculate the Intersection over Union (IoU) or overlap
double TrackerPerformanceEvaluator::calculateOverlap(const cv::Rect& groundTruth, const cv::Rect& trackingResult)
{
  int intersectionArea = (groundTruth & trackingResult).area();
  int unionArea = groundTruth.area() + trackingResult.area() - intersectionArea;
  return static_cast<double>(intersectionArea) / unionArea;
}

// Private helper method to calculate the center localization error
double TrackerPerformanceEvaluator::calculateCenterError(const cv::Rect& groundTruth, const cv::Rect& trackingResult)
{
  cv::Point2f gtCenter = (groundTruth.tl() + groundTruth.br()) * 0.5;
  cv::Point2f trCenter = (trackingResult.tl() + trackingResult.br()) * 0.5;
  return cv::norm(gtCenter - trCenter);
}

// Method to add a single frame's results
void TrackerPerformanceEvaluator::addFrameResult(const cv::Rect& groundTruth, const cv::Rect& trackingResult,
                                                 double processing_time)
{
  overlaps.push_back(calculateOverlap(groundTruth, trackingResult));
  errors.push_back(calculateCenterError(groundTruth, trackingResult));
  processing_times.push_back(processing_time);
}

// Method to calculate and return the average overlap
double TrackerPerformanceEvaluator::getAverageOverlap() const
{
  if (overlaps.empty())
    return 0.0;
  double sum = std::accumulate(overlaps.begin(), overlaps.end(), 0.0);
  return sum / overlaps.size();
}

// Method to calculate and return the average center error
double TrackerPerformanceEvaluator::getAverageError() const
{
  if (errors.empty())
    return 0.0;
  double sum = std::accumulate(errors.begin(), errors.end(), 0.0);
  return sum / errors.size();
}

// Method to save the results to a file
void TrackerPerformanceEvaluator::saveResultsToFile(const std::string& filename) const
{
  std::ofstream file(filename);
  if (!file.is_open())
  {
    std::cerr << "Could not open the file: " << filename << std::endl;
    return;
  }

  file << "Frame,Overlap,Center Error,Processing Time\n";
  for (size_t i = 0; i < overlaps.size(); ++i)
  {
    file << i + 1 << "," << overlaps[i] << "," << errors[i] << "," << processing_times[i] << "\n";
  }

  file << "\nAverage Overlap: " << getAverageOverlap() << "\n";
  file << "Average Center Error: " << getAverageError() << "\n";

  file.close();
}
