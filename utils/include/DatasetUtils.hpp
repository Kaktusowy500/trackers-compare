#pragma once
#include <iostream>
#include <filesystem>
#include <vector>

enum class DatasetType
{
    Custom,
    OTB,
    Unknown
};

struct DatasetInfo
{
    std::string media_path;
    DatasetType dataset_type;
    std::vector<std::string> ground_truth_paths;
};

std::ostream &operator<<(std::ostream &os, const DatasetInfo &datasetInfo);
DatasetInfo getDatasetInfo(const std::string &path);
