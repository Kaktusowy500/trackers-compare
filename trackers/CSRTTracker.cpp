#include "CSRTTracker.hpp"

CSRTTracker::CSRTTracker()
{
    name = "CSRT";
    tracker = cv::TrackerCSRT::create();
}

CSRTTracker::~CSRTTracker() {}

void CSRTTracker::init(const cv::Mat& frame, const cv::Rect& roi)
{
    tracker->init(frame, roi);
    setState(TrackerState::Tracking);
}

bool CSRTTracker::update(const cv::Mat& frame, cv::Rect& roi)
{
    bool ok = tracker->update(frame, roi);
    TrackerState state_to_set = ok ? TrackerState::Tracking : TrackerState::Recovering;
    setState(state_to_set);
    return ok;
}