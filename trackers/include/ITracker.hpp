#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

class ITracker
{
public:
    virtual ~ITracker() {}
    virtual void init(const cv::Mat &frame, cv::Rect roi) = 0;
    virtual bool update(const cv::Mat &frame, cv::Rect& roi) = 0;
protected:
    cv::Ptr<cv::Tracker> tracker;
};