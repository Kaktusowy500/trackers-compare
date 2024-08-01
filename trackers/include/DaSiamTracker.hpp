#pragma once
#include "ITracker.hpp"

class DaSiamTracker : public ITracker
{
public:
    DaSiamTracker();
    ~DaSiamTracker();
    virtual void init(const cv::Mat &frame, const cv::Rect &roi);
    virtual bool update(const cv::Mat &frame, cv::Rect &roi);
    double getTrackingScore();
private:
    double scoreThresh;
};