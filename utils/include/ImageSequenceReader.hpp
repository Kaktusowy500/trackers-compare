#pragma once
#include <filesystem>
#include <algorithm>
#include "VideoReader.hpp"

namespace fs = std::filesystem;

class ImageSequenceReader : public VideoReader
{
private:
    std::vector<std::string> imageFiles;
    size_t currentIndex;
    bool done;

public:
    ImageSequenceReader(const std::string &directoryPath) : currentIndex(0), done(false)
    {
        for (const auto &entry : fs::directory_iterator(directoryPath))
        {
            if (entry.is_regular_file())
            {
                std::string ext = entry.path().extension().string();
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png")
                {
                    imageFiles.push_back(entry.path().string());
                }
            }
        }

        std::sort(imageFiles.begin(), imageFiles.end());

        if (imageFiles.empty())
        {
            std::cerr << "No images found in directory: " << directoryPath << std::endl;
            done = true;
        }
    }

    bool getNextFrame(cv::Mat &frame) override
    {
        if (done || currentIndex >= imageFiles.size())
        {
            done = true;
            return false;
        }

        frame = cv::imread(imageFiles[currentIndex]);
        if (frame.empty())
        {
            std::cerr << "Failed to load image: " << imageFiles[currentIndex] << std::endl;
            done = true;
            return false;
        }

        currentIndex++;
        return true;
    }

    bool isDone() const override
    {
        return done;
    }

    void reset() override
    {
        currentIndex = 0;
        done = false;
    }
};
