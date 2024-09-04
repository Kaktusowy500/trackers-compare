#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

enum class TrackerState
{
    Ready,
    Tracking,
    Recovering, // Tracking score is too low, but algo is conscious about it and tries to recover
    Lost, // Totaly lost, no hope to recover
    ToBeReinited // Tracker is lost, but it is to be reinited
};
std::string stateToString(TrackerState state);


class ITracker
{
public:
    ITracker();
    virtual ~ITracker() {}

    virtual void init(const cv::Mat& frame, const cv::Rect& roi) = 0;
    virtual bool update(const cv::Mat& frame, cv::Rect& roi) = 0;
    virtual double getTrackingScore();
    std::string getName();
    TrackerState getState();
    void setState(TrackerState s);


protected:
    cv::Ptr<cv::Tracker> tracker;
    std::string name;
    TrackerState state;
};