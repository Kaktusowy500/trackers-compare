#pragma once
#include "ITracker.hpp"

class VITTracker : public ITracker
{
public:
    VITTracker();
    ~VITTracker();
    virtual void init(const cv::Mat &frame, cv::Rect roi);
    virtual bool update(const cv::Mat &frame, cv::Rect &roi);
};