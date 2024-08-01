#include "VITTracker.hpp"

VITTracker::VITTracker()
{
    name = "VIT";
    cv::TrackerVit::Params params;
    params.net = "nn_models/vit.onnx";
    tracker = cv::TrackerVit::create(params);
}
VITTracker::~VITTracker() {}

void VITTracker::init(const cv::Mat &frame, cv::Rect roi)
{
    tracker->init(frame, roi);
}

bool VITTracker::update(const cv::Mat &frame, cv::Rect &roi)
{
    bool ok = tracker->update(frame, roi);
    return ok;
}