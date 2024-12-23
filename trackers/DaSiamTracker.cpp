#include "DaSiamTracker.hpp"

DaSiamTracker::DaSiamTracker(double score_thresh): score_thresh(score_thresh)
{
    name = "DaSiam";
    cv::TrackerDaSiamRPN::Params params;
    params.model = "nn_models/dasiamrpn_model.onnx";
    params.kernel_cls1 = "nn_models/dasiamrpn_kernel_cls1.onnx";
    params.kernel_r1 = "nn_models/dasiamrpn_kernel_r1.onnx";
    tracker = cv::TrackerDaSiamRPN::create(params);
}

DaSiamTracker::~DaSiamTracker() {}

void DaSiamTracker::init(const cv::Mat& frame, const cv::Rect& roi)
{
    tracker->init(frame, roi);
    setState(TrackerState::Tracking);
}

bool DaSiamTracker::update(const cv::Mat& frame, cv::Rect& roi)
{
    bool ok = tracker->update(frame, roi);
    double score = getTrackingScore();
    TrackerState state_to_set = score >= score_thresh ? TrackerState::Tracking : TrackerState::Recovering;
    setState(state_to_set);

    return ok;
}

double DaSiamTracker::getTrackingScore()
{
    return tracker.dynamicCast<cv::TrackerDaSiamRPN>()->getTrackingScore();
}