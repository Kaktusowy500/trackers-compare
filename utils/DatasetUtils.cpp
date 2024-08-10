#include "DatasetUtils.hpp"

namespace fs = std::filesystem;

std::ostream &operator<<(std::ostream &os, const DatasetInfo &datasetInfo)
{
    os << "Media Path: " << datasetInfo.media_path << '\n';
    os << "Dataset Type: " << static_cast<int>(datasetInfo.dataset_type) << '\n';
    os << "Ground Truth Paths: \n";

    for (const auto &path : datasetInfo.ground_truth_paths)
    {
        os << "  - " << path << '\n';
    }

    return os;
}

DatasetInfo getDatasetInfo(const std::string &path)
{
    DatasetInfo dataset_info;
    if (fs::is_directory(path))
    {
        for (const auto &entry : fs::directory_iterator(path))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".mp4")
            {
                dataset_info.media_path = entry.path().string();
                dataset_info.dataset_type = DatasetType::Custom;
            }
            else if (entry.is_directory() && entry.path().filename() == "img")
            {
                dataset_info.media_path = entry.path().string();
                dataset_info.dataset_type = DatasetType::OTB;
            }
            else if (entry.is_regular_file() && entry.path().extension() == ".txt")
            {
                dataset_info.ground_truth_paths.push_back(entry.path().string());
            }
        };
    }
    else
    {
        std::cerr << "The provided path is not a directory." << std::endl;
    }
    return dataset_info;
}