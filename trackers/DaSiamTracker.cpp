#include "DaSiamTracker.hpp"

DaSiamTracker::DaSiamTracker()
{
    name = "DaSiam";
    cv::TrackerDaSiamRPN::Params params;
    params.model = "nn_models/dasiamrpn_model.onnx";
    params.kernel_cls1 = "nn_models/dasiamrpn_kernel_cls1.onnx";
    params.kernel_r1 = "nn_models/dasiamrpn_kernel_r1.onnx";
    tracker = cv::TrackerDaSiamRPN::create(params);
    scoreThresh = 0.9;
}

DaSiamTracker::~DaSiamTracker() {}

void DaSiamTracker::init(const cv::Mat &frame, const cv::Rect &roi)
{
    tracker->init(frame, roi);
    state = TrackerState::Tracking;
}

bool DaSiamTracker::update(const cv::Mat &frame, cv::Rect &roi)
{
    bool ok = tracker->update(frame, roi);
    double score = getTrackingScore();
    if (score < scoreThresh)
    {
        state = TrackerState::Lost;
    }
    
    return ok;
}

double DaSiamTracker::getTrackingScore()
{
    return tracker.dynamicCast<cv::TrackerDaSiamRPN>()->getTrackingScore();
}