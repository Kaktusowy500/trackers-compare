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

struct Annotation
{
    cv::Rect2f rect;   
    int frame;         
    int occluded = -1; // Occlusion status, default is -1 indicating unknown
};


std::ostream &operator<<(std::ostream &os, const DatasetInfo &datasetInfo);
DatasetInfo getDatasetInfo(const std::string &path);
std::vector<Annotation> loadOTBAnnotations(const std::string& filename);
std::vector<Annotation> loadCustomAnnotations(const std::string& filename);


