#include "TrackerModVIT.hpp"
#include "ModVITTracker.hpp"

ModVITTracker::ModVITTracker(double score_thresh) :score_thresh(score_thresh)
{
    name = "ModVIT";
    cv::TrackerModVIT::Params params;
    params.net = "nn_models/vit.onnx";
    tracker = cv::TrackerModVIT::create(params);
}
ModVITTracker::~ModVITTracker() {}

void ModVITTracker::init(const cv::Mat& frame, const cv::Rect& roi)
{
    tracker->init(frame, roi);
    setState(TrackerState::Tracking);
}

bool ModVITTracker::update(const cv::Mat& frame, cv::Rect& roi)
{
    bool ok = tracker->update(frame, roi);
    double score = getTrackingScore();
    TrackerState state_to_set = score >= score_thresh ? TrackerState::Tracking : TrackerState::Recovering;
    setState(state_to_set);
    return ok;
}

double ModVITTracker::getTrackingScore()
{
    return tracker.dynamicCast<cv::TrackerModVIT>()->getTrackingScore();
}