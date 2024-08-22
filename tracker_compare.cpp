#include <iostream>
#include <memory>
#include <random>
#include <filesystem>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include "DatasetUtils.hpp"
#include "VideoFileReader.hpp"
#include "ImageSequenceReader.hpp"
#include "ITracker.hpp"
#include "CSRTTracker.hpp"
#include "VITTracker.hpp"
#include "DaSiamTracker.hpp"

std::vector<cv::Rect> loadRectsFromFile(const std::string &filename)
{
    // Each row in the ground-truth files represents the bounding box of the target in that frame,
    // (x, y, box-width, box-height).

    std::vector<cv::Rect> rects;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Could not open the file: " << filename << std::endl;
        return rects;
    }

    std::string line;
    while (std::getline(file, line))
    {
        // Replace commas with spaces to handle comma-separated values
        std::replace(line.begin(), line.end(), ',', ' ');

        std::istringstream iss(line);
        int x, y, width, height;
        if (iss >> x >> y >> width >> height)
        {
            rects.emplace_back(x, y, width, height);
        }
    }

    file.close();
    return rects;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << "clip directory" << std::endl;
        return -1;
    }

    DatasetInfo dataset_info = getDatasetInfo(argv[1]);
    std::cout << dataset_info << std::endl;
    auto ground_truths = loadRectsFromFile(dataset_info.ground_truth_paths[0]);
    std::cout << "Ground truth vector size: " << ground_truths.size() << std::endl;

    std::unique_ptr<VideoReader> videoReader;
    if (dataset_info.dataset_type == DatasetType::OTB)
    {
        videoReader = std::make_unique<ImageSequenceReader>(dataset_info.media_path);
    }
    else if (dataset_info.dataset_type == DatasetType::Custom)
    {
        videoReader = std::make_unique<VideoFileReader>(dataset_info.media_path);
    }
    else
    {
        std::cout << "Unknown dataset type." << std::endl;
        return -1;
    }

    std::vector<std::unique_ptr<ITracker>> trackers;
    trackers.push_back(std::make_unique<CSRTTracker>());
    trackers.push_back(std::make_unique<VITTracker>());
    trackers.push_back(std::make_unique<DaSiamTracker>());

    std::mt19937 rng;
    rng.seed(1234567890);
    std::uniform_int_distribution<std::mt19937::result_type> dist255(0, 255);
    std::vector<cv::Scalar> colors;
    for (const auto &t : trackers)
    {
        colors.push_back(cv::Scalar(dist255(rng), dist255(rng), dist255(rng)));
    }


    cv::Mat frame;
    int frame_count = 0;
    if (videoReader->getNextFrame(frame))
    {
        for (auto &t : trackers)
        {
            t->init(frame, ground_truths[frame_count]);
        }
        frame_count++;
    }
    else
    {
        std::cout << "Error reading first frame." << std::endl;
        return -1;
    }
    cv::rectangle(frame, ground_truths[frame_count], cv::Scalar(0, 0, 255), 2, 1);
    cv::imshow("First frame", frame);

    while (!videoReader->isDone())
    {
        if (videoReader->getNextFrame(frame))
        {
            cv::rectangle(frame, ground_truths[frame_count], cv::Scalar(0, 0, 255), 2, 1);

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

            cv::imshow("Frame", frame);
            if (cv::waitKey(30) >= 0)
                break; // Press any key to exit
            frame_count++;
        }
    }

    return 0;
}
