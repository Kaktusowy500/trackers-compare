#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include "DatasetUtils.hpp"
#include "VideoReader.hpp"
#include "ITracker.hpp"
#include "TrackerPerformanceEvaluator.hpp"



class TrackerComparator
{
public:
    void loadDataset(std::string path);
    bool setupComponents();
    void runEvaluation();
    void saveResults(std::string path);
private:
    bool readFirstFrameAndInit();
    bool setupVideoReader();
    bool setupTrackersAndEvaluators();
    
    DatasetInfo dataset_info;
    std::unique_ptr<VideoReader> videoReader;
    std::vector<cv::Rect> ground_truths;
    std::vector<std::unique_ptr<ITracker>> trackers;
    std::vector<std::unique_ptr<TrackerPerformanceEvaluator>> evaluators;
    std::vector<cv::Scalar> colors;
    cv::Mat frame;
    unsigned int frame_count = 0;

};


