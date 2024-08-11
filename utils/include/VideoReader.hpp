#pragma once

#include <opencv2/opencv.hpp>

class VideoReader
{
public:
    virtual ~VideoReader() = default;

    virtual bool getNextFrame(cv::Mat &frame) = 0;

    virtual bool isDone() const = 0;

    virtual void reset() = 0;
};
