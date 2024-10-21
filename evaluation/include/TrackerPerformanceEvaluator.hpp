#pragma once

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <vector>
#include <string>
#include "SequenceTrackingSummary.hpp"

struct FrameResult
{
    double overlap = -1.0 ;         // percentage of overlap between the ground truth and tracking result
    double error = -1.0;           // localization errors for each frame in pixels
    double processing_time = -1.0; // processing times for each frame in seconds
    double bbox_area = -1.0;       // area of the bounding box in pixels
    bool valid = false;             // whether the tracking result is valid or not
};

enum class ValidationStatus
{
    Valid,
    NonValidTrackerLost,
    NonValidOverlap,
    NonValidCenterError
};

std::string ValidationStatusToString(ValidationStatus status);

struct TrackerPerformanceEvaluatorArgs
{
    std::string tracker_name;
    double overlap_thresh = 0.3;
    double center_error_thresh = 0.3;
};

class TrackerPerformanceEvaluator
{
public:
    TrackerPerformanceEvaluator(const TrackerPerformanceEvaluatorArgs& args);
    ValidationStatus validateAndAddResult(const cv::Rect& ground_truth, const cv::Rect& tracking_result, double processing_time, bool prior_valid);

    double getAverageOverlap() const;
    double getAverageError() const;
    double getAverageProcessingTime() const;
    double getValidFramePercent() const;
    double getOverlapStd() const;
    double getErrorStd() const;
    double getProcessingTimeStd() const;

    void saveResultsToFile(const std::string& filename) const;

    SequenceTrackingSummary getTrackingSummary();

    void trackingReinited()
    {
        spdlog::info("Tracker: {} reinited", tracker_name);
        reinit_cnt++;
    }
    unsigned int getReinitCount()
    {
        return reinit_cnt;
    }

private:
    double calculateOverlap(const cv::Rect& ground_truth, const cv::Rect& tracking_result);
    double calculateCenterError(const cv::Rect& ground_truth, const cv::Rect& tracking_result);

    std::vector<FrameResult> results;
    std::string tracker_name;
    // params loaded from config
    double overlap_thresh = 0.3;
    double center_error_thresh = 0.3; // normalized diagonal of the bounding box
    
    unsigned int reinit_cnt = 0;
};
