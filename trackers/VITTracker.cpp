#include "VITTracker.hpp"

VITTracker::VITTracker()
{
    name = "VIT";
    cv::TrackerVit::Params params;
    params.net = "nn_models/vit.onnx";
    tracker = cv::TrackerVit::create(params);
    scoreThresh = 0.4;
}
VITTracker::~VITTracker() {}

void VITTracker::init(const cv::Mat &frame, const cv::Rect &roi)
{
    tracker->init(frame, roi);
    state = TrackerState::Tracking;
}

bool VITTracker::update(const cv::Mat &frame, cv::Rect &roi)
{
    bool ok = tracker->update(frame, roi);
    double score = getTrackingScore();
    if (score < scoreThresh)
    {
        state = TrackerState::Lost;
    }
    return ok;
}

double VITTracker::getTrackingScore()
{
    return tracker.dynamicCast<cv::TrackerVit>()->getTrackingScore();
}