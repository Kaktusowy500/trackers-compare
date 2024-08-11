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

    cv::Mat frame;
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

    int frame_count = 0;
    while (!videoReader->isDone())
    {
        if (videoReader->getNextFrame(frame))
        {
            cv::rectangle(frame, ground_truths[frame_count], cv::Scalar(0, 0, 255), 2, 1);
            cv::imshow("Frame", frame);
            if (cv::waitKey(30) >= 0)
                break; // Press any key to exit
            frame_count++;
        }
    }

    return 0;
}
