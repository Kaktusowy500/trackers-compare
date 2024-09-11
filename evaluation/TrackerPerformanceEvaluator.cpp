#include "TrackerPerformanceEvaluator.hpp"
#include <fstream>
#include <numeric>
#include <iostream>

std::string ValidationStatusToString(ValidationStatus status)
{
  static std::map<ValidationStatus, std::string> map = {
      {ValidationStatus::Valid, "Valid"},
      {ValidationStatus::NonValidTrackerLost, "NonValidTrackerLost"},
      {ValidationStatus::NonValidOverlap, "NonValidOverlap"},
      {ValidationStatus::NonValidCenterError, "NonValidCenterError"}
  };
  return map[status];
}

TrackerPerformanceEvaluator::TrackerPerformanceEvaluator(const TrackerPerformanceEvaluatorArgs& args)
{
  tracker_name = args.tracker_name;
  overlap_thresh = args.overlap_thresh;
  center_error_thresh = args.center_error_thresh;
}

// Private helper method to calculate the Intersection over Union (IoU) or overlap
double TrackerPerformanceEvaluator::calculateOverlap(const cv::Rect& ground_truth, const cv::Rect& tracking_result)
{
  int intersection_area = (ground_truth & tracking_result).area();
  int union_area = ground_truth.area() + tracking_result.area() - intersection_area;
  return static_cast<double>(intersection_area) / union_area;
}

// Private helper method to calculate the center localization error
double TrackerPerformanceEvaluator::calculateCenterError(const cv::Rect& ground_truth, const cv::Rect& tracking_result)
{
  cv::Point2f gt_center = (ground_truth.tl() + ground_truth.br()) * 0.5;
  cv::Point2f tr_center = (tracking_result.tl() + tracking_result.br()) * 0.5;
  return cv::norm(gt_center - tr_center);
}

// Method to add a single frame's results
ValidationStatus TrackerPerformanceEvaluator::validateAndAddResult(const cv::Rect& ground_truth, const cv::Rect& tracking_result,
  double processing_time, bool trackerLost)
{
  FrameResult result;
  ValidationStatus valid_status = !trackerLost ? ValidationStatus::Valid : ValidationStatus::NonValidTrackerLost;
  if (valid_status == ValidationStatus::Valid)
  {
    result.overlap = calculateOverlap(ground_truth, tracking_result);
    result.error = calculateCenterError(ground_truth, tracking_result);
    result.processing_time = processing_time;
    result.bbox_area = tracking_result.area();

    double diagonal = std::sqrt(std::pow(ground_truth.width, 2) + std::pow(ground_truth.height, 2));
    double normalized_centre_error = result.error / diagonal;
    if (result.overlap < overlap_thresh)
      valid_status = ValidationStatus::NonValidOverlap;
    if (normalized_centre_error > center_error_thresh)
      valid_status = ValidationStatus::NonValidCenterError;
  }
  result.valid = (valid_status == ValidationStatus::Valid);
  results.push_back(result);
  return valid_status;
}

// Method to calculate and return the average overlap
double TrackerPerformanceEvaluator::getAverageOverlap() const
{
  double sum_overlap = 0.0;
  int valid_count = 0;

  for (const auto& result : results)
  {
    if (result.valid)
    {
      sum_overlap += result.overlap;
      valid_count++;
    }
  }

  return valid_count > 0 ? sum_overlap / valid_count : 0.0;
}

// Method to calculate and return the average center error
double TrackerPerformanceEvaluator::getAverageError() const
{
  double sum_error = 0.0;
  int valid_count = 0;

  for (const auto& result : results)
  {
    if (result.valid)
    {
      sum_error += result.error;
      valid_count++;
    }
  }

  return valid_count > 0 ? sum_error / valid_count : 0.0;
}

double TrackerPerformanceEvaluator::getAverageProcessingTime() const
{
  double sum_processing_time = 0.0;
  int valid_count = 0;

  for (const auto& result : results)
  {
    if (result.valid)
    {
      sum_processing_time += result.processing_time;
      valid_count++;
    }
  }

  return valid_count > 0 ? sum_processing_time / valid_count : 0.0;
}

double TrackerPerformanceEvaluator::getValidFramePercent() const
{
  int valid_count = 0;

  for (const auto& result : results)
  {
    if (result.valid)
      valid_count++;
  }
  return valid_count / static_cast<double>(results.size());
}

// Method to save the results to a file
void TrackerPerformanceEvaluator::saveResultsToFile(const std::string& filename) const
{
  std::ofstream file(filename);
  if (!file.is_open())
  {
    spdlog::info("Could not open the file: {}", filename);
    return;
  }

  file << "Frame,Overlap,Center Error,Processing Time,BBox Area,Valid" << std::endl;
  for (size_t i = 0; i < results.size(); ++i)
  {
    file << i + 1 << "," << results[i].overlap << "," << results[i].error << "," << results[i].processing_time << "," << results[i].bbox_area << "," << results[i].valid << "\n";
  }

  file.close();
}

SequenceTrackingSummary TrackerPerformanceEvaluator::getTrackingSummary()
{
  SequenceTrackingSummary summary;
  summary.average_overlap = getAverageOverlap();
  summary.average_error = getAverageError();
  summary.average_processing_time = getAverageProcessingTime();
  summary.valid_frame_percent = getValidFramePercent();
  summary.reinit_count = reinit_count;


  spdlog::info("Tracker: {} statistics:\n"
    "Average Overlap: {}\n"
    "Average Center Error: {}\n"
    "Average Processing Time: {}\n"
    "Valid Time Tracking Percentage: {}\n"
    "Reinit number: {}",
    tracker_name,
    summary.average_overlap,
    summary.average_error,
    summary.average_processing_time,
    summary.valid_frame_percent,
    summary.reinit_count);

  return summary;
}