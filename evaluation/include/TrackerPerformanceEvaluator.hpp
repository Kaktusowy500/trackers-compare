#pragma once

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <vector>
#include <string>

struct FrameResult
{
    double overlap;         // percentage of overlap between the ground truth and tracking result
    double error;           // localization errors for each frame in pixels
    double processing_time; // processing times for each frame in seconds
    double bbox_area;       // area of the bounding box in pixels
    bool valid;             // whether the tracking result is valid or not
};

class TrackerPerformanceEvaluator
{
public:
    TrackerPerformanceEvaluator(const std::string &tracker_name);
    // Method to add a single frame's results
    bool addFrameResult(const cv::Rect &groundTruth, const cv::Rect &trackingResult, double processing_time, bool valid = true);

    // Method to calculate and return the average overlap
    double getAverageOverlap() const;

    // Method to calculate and return the average center error
    double getAverageError() const;

    double getAverageProcessingTime() const;

    // Method to save the results to a file
    void saveResultsToFile(const std::string &filename) const;
    void trackingReinited()
    {
        spdlog::debug("Tracker {} reinited", tracker_name);
        reinit_count++;
    }
    unsigned int getReinitCount()
    {
        return reinit_count;
    }

private:
    double calculateOverlap(const cv::Rect &groundTruth, const cv::Rect &trackingResult);
    double calculateCenterError(const cv::Rect &groundTruth, const cv::Rect &trackingResult);

    std::vector<FrameResult> results;
    std::string tracker_name;
    // params
    double overlapThresh = 0.4;
    double centerErrorThresh = 0.2; // normalized diagonal of the bounding box
    unsigned int reinit_count = 0;
};
