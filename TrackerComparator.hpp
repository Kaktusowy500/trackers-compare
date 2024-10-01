#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
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
    TrackerComparator(const YAML::Node& config);
    ~TrackerComparator();
    void loadDataset(const DatasetInfo& d_info);
    void loadVideoOnlyDataset(const std::string& path);
    bool setupComponents(const std::string & instance_results_dir = "");
    void runEvaluation();
    void runPreview(const std::string & tracker_name);
    void saveResults(const std::string & path);
    void reset();
private:
    bool readFirstFrameAndInit();
    bool setupVideoReader();
    bool setupTrackersAndEvaluators();
    void setupVideoWriter(const std::string& instance_results_dir);
    void convertGTToNonNormalized(int imgWidth, int imgHeight);
    void parseReinitStrategy(const std::string& strategy);
    bool applyReinitStrategy(const cv::Mat& frame, int index, ValidationStatus valid_status);
    unsigned calcWaitTime();

    DatasetInfo dataset_info;
    std::unique_ptr<VideoReader> video_reader;
    cv::VideoWriter video_writer;
    std::vector<Annotation> ground_truths;
    std::vector<std::unique_ptr<ITracker>> trackers;
    std::vector<std::unique_ptr<TrackerPerformanceEvaluator>> evaluators;
    std::vector<cv::Scalar> colors;
    cv::Mat frame;
    std::chrono::time_point<std::chrono::steady_clock> start_frame_processing_time;
    unsigned int desired_frame_processing_time = 0;
    unsigned int frame_count = 0;

    const YAML::Node& config;
    ReinitStrategy reinit_strategy;

};


