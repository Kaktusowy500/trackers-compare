#pragma once
#include "ITracker.hpp"

class CSRTTracker : public ITracker
{
public:
    CSRTTracker();
    ~CSRTTracker();
    virtual void init(const cv::Mat &frame, const cv::Rect &roi);
    virtual bool update(const cv::Mat &frame, cv::Rect &roi);
};