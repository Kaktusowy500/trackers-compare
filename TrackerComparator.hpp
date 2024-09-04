#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include "DatasetUtils.hpp"
#include "VideoReader.hpp"
#include "ITracker.hpp"
#include "TrackerPerformanceEvaluator.hpp"

// Compare strategies
// Reset imidiately after loss, count resets and avg tracking time
// Reset after some time to allow recovery
// One init and do not count when aim was lost

// 2 validity info, one from tracker itself, one from evaluator

enum class ReinitStrategy
{
    Immediate,
    Delayed,
    OneInit
};


class TrackerComparator
{
public:
    TrackerComparator();
    void loadDataset(std::string path, bool only_video = false);
    bool setupComponents();
    void runEvaluation();
    void runPreview(std::string tracker_name);
    void saveResults(std::string path);
private:
    bool readFirstFrameAndInit();
    bool setupVideoReader();
    bool setupTrackersAndEvaluators();
    void convertGTToNonNormalized(int imgWidth, int imgHeight);
    void applyReinitStrategy(const cv::Mat& frame, int index, ValidationStatus valid_status);

    DatasetInfo dataset_info;
    std::unique_ptr<VideoReader> video_reader;
    std::vector<Annotation> ground_truths;
    std::vector<std::unique_ptr<ITracker>> trackers;
    std::vector<std::unique_ptr<TrackerPerformanceEvaluator>> evaluators;
    std::vector<cv::Scalar> colors;
    cv::Mat frame;
    ReinitStrategy reinit_strategy;
    unsigned int frame_count = 0;

};


