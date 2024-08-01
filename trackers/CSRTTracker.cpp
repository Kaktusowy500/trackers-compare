#include "CSRTTracker.hpp"

CSRTTracker::CSRTTracker()
{
    name = "CSRT";
    tracker = cv::TrackerCSRT::create();
}

CSRTTracker::~CSRTTracker() {}

void CSRTTracker::init(const cv::Mat &frame, cv::Rect roi)
{
    tracker->init(frame, roi);
}

bool CSRTTracker::update(const cv::Mat &frame, cv::Rect &roi)
{
    bool ok = tracker->update(frame, roi);
    return ok;
}