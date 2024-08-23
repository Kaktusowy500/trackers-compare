#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class TrackerPerformanceEvaluator {
private:
    std::vector<double> overlaps;  // percentage of overlap between the ground truth and tracking result
    std::vector<double> errors;    // localization errors for each frame in pixels
    std::vector<double> processing_times;    // processing times for each frame in seconds

    double calculateOverlap(const cv::Rect &groundTruth, const cv::Rect &trackingResult);
    double calculateCenterError(const cv::Rect &groundTruth, const cv::Rect &trackingResult);

public:
    // Method to add a single frame's results
    void addFrameResult(const cv::Rect &groundTruth, const cv::Rect &trackingResult, double processing_time);

    // Method to calculate and return the average overlap
    double getAverageOverlap() const;

    // Method to calculate and return the average center error
    double getAverageError() const;

    // Method to save the results to a file
    void saveResultsToFile(const std::string &filename) const;
};

