#pragma once
#include <iostream>
#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>

enum class DatasetType
{
    Custom,
    OTB,
    Unknown
};

struct DatasetInfo
{
    std::string name;
    std::string media_path;
    DatasetType dataset_type;
    std::vector<std::string> ground_truth_paths;
};

std::ostream &operator<<(std::ostream &os, const DatasetInfo &datasetInfo);
DatasetInfo getDatasetInfo(const std::string &path);
std::vector<cv::Rect> loadRectsFromFile(const std::string& filename);

