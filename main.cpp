#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <video_file>" << std::endl;
        return -1;
    }

    cv::VideoCapture video(argv[1]);
    if (!video.isOpened()) {
        std::cout << "Error opening video file." << std::endl;
        return -1;
    }

    cv::Mat frame;
    video >> frame;
    if (frame.empty()) {
        std::cout << "Error reading first frame." << std::endl;
        return -1;
    }

    cv::Rect2d bbox = cv::selectROI("Tracking", frame, false);

    cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();

    tracker->init(frame, bbox);

    while (video.read(frame)) {
        cv::Rect2d bbox;
        bool ok = tracker->update(frame, bbox);
        cv::Scalar color = ok ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::putText(frame, "CSRT", cv::Point(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        cv::rectangle(frame, bbox, color, 2, 1);

        cv::imshow("Tracking", frame);
        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}
