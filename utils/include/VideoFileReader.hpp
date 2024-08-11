#pragma once
#include "VideoReader.hpp"

class VideoFileReader : public VideoReader
{
private:
    cv::VideoCapture video;
    bool done;

public:
    VideoFileReader(const std::string &videoPath) : done(false)
    {
        video.open(videoPath);
        if (!video.isOpened())
        {
            std::cerr << "Failed to open video file: " << videoPath << std::endl;
            done = true;
        }
    }

    bool getNextFrame(cv::Mat &frame) override
    {
        if (done)
            return false;

        if (!video.read(frame))
        {
            done = true;
            return false;
        }
        return true;
    }

    bool isDone() const override
    {
        return done;
    }

    void reset() override
    {
        video.set(cv::CAP_PROP_POS_FRAMES, 0);
        done = false;
    }
};
