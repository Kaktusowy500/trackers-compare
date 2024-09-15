#pragma once
#include "ITracker.hpp"

class ModVITTracker : public ITracker
{
public:
    ModVITTracker(double score_thresh);
    ~ModVITTracker();
    virtual void init(const cv::Mat &frame, const cv::Rect &roi);
    virtual bool update(const cv::Mat &frame, cv::Rect &roi);
    virtual double getTrackingScore();
private:
    double score_thresh;
};