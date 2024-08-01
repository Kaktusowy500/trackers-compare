#include "CSRTTracker.hpp"

CSRTTracker::CSRTTracker()
{
    name = "CSRT";
    tracker = cv::TrackerCSRT::create();
}

CSRTTracker::~CSRTTracker() {}

void CSRTTracker::init(const cv::Mat &frame, const cv::Rect &roi)
{
    tracker->init(frame, roi);
    state = TrackerState::Tracking;
}

bool CSRTTracker::update(const cv::Mat &frame, cv::Rect &roi)
{
    bool ok = tracker->update(frame, roi);
    if (!ok)
    {
        state = TrackerState::Lost;
    }
    return ok;
}