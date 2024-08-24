#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

enum class TrackerState
{
    Ready,
    Tracking,
    Recovering,
    Lost,
    ToBeReinited
};
std::string stateToString(TrackerState state);


class ITracker
{
public:
    ITracker();
    virtual ~ITracker() {}

    virtual void init(const cv::Mat &frame, const cv::Rect &roi) = 0;
    virtual bool update(const cv::Mat &frame, cv::Rect &roi) = 0;
    virtual double getTrackingScore();
    std::string getName();
    TrackerState getState();


protected:
    cv::Ptr<cv::Tracker> tracker;
    std::string name;
    TrackerState state;
};