#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <memory>
#include <random>
#include "ITracker.hpp"
#include "CSRTTracker.hpp"
#include "VITTracker.hpp"
#include "DaSiamTracker.hpp"

int main(int argc, char **argv)
{
    std::cout << "OpenCV version: " << cv::getVersionString() << std::endl;
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <video_file>" << std::endl;
        return -1;
    }

    cv::VideoCapture video(argv[1]);
    if (!video.isOpened())
    {
        std::cout << "Error opening video file." << std::endl;
        return -1;
    }

    cv::Mat frame;
    video >> frame;
    if (frame.empty())
    {
        std::cout << "Error reading first frame." << std::endl;
        return -1;
    }

    std::vector<std::unique_ptr<ITracker>> trackers;
    trackers.push_back(std::make_unique<CSRTTracker>());
    trackers.push_back(std::make_unique<VITTracker>());
    trackers.push_back(std::make_unique<DaSiamTracker>());

    cv::Rect bbox = cv::selectROI("Tracking", frame, false);
    for (auto &t : trackers)
    {
        t->init(frame, bbox);
    }

    std::mt19937 rng;
    rng.seed(1234567890);
    std::uniform_int_distribution<std::mt19937::result_type> dist255(0, 255);
    std::vector<cv::Scalar> colors;
    for (auto &t : trackers)
    {
        colors.push_back(cv::Scalar(dist255(rng), dist255(rng), dist255(rng)));
    }

    while (video.read(frame))
    {
        for (int i = 0; i < trackers.size(); i++)
        {
            cv::Rect bbox;
            trackers[i]->update(frame, bbox);
            bool trackingValid = (trackers[i]->getState() == TrackerState::Tracking);
            // if (trackingValid)
            // {

            auto color = trackingValid ? colors[i] : cv::Scalar(0, 0, 255);
            cv::putText(frame, trackers[i]->getName(), cv::Point(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            cv::rectangle(frame, bbox, color, 2, 1);
            // }
            std::string state_str = stateToString(trackers[i]->getState());
            cv::Scalar state_color = trackingValid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::putText(frame, trackers[i]->getName() + " : " + state_str + " " + std::to_string(trackers[i]->getTrackingScore()), cv::Point(10, (frame.rows - 20) - 30 * i),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2);
        }

        cv::imshow("Tracking", frame);
        if (cv::waitKey(1) == 27)
            break;
    }

    return 0;
}
