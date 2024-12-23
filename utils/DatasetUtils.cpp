#include "DatasetUtils.hpp"
#include <fstream>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

std::ostream& operator<<(std::ostream& os, const DatasetInfo& datasetInfo)
{
    os << "Name: " << datasetInfo.name << '\n';
    os << "Media Path: " << datasetInfo.media_path << '\n';
    os << "Dataset Type: " << static_cast<int>(datasetInfo.dataset_type) << '\n';
    os << "Ground Truth Paths: \n";

    for (const auto& path : datasetInfo.ground_truth_paths)
    {
        os << "  - " << path << '\n';
    }

    return os;
}

std::vector<DatasetInfo> loadDatasetInfos(const std::string& path)
{
    std::vector<DatasetInfo> dataset_infos;
    auto datasets_paths = getAllDirectories(path);
    if(datasets_paths.empty())
    {
        spdlog::debug("No directories in provided path, assuming it is path to dataset instance");
        DatasetInfo dataset_info = getDatasetInfo(path);
        if (dataset_info.dataset_type != DatasetType::Unknown)
        {
            dataset_infos.push_back(dataset_info);
        }
    }
    else
    {
        for (const auto& dataset_path : datasets_paths)
        {
            DatasetInfo dataset_info = getDatasetInfo(dataset_path.string());
            if (dataset_info.dataset_type != DatasetType::Unknown)
            {
                dataset_infos.push_back(dataset_info);
            }
        }
    }
    spdlog::debug("Loaded {} dataset infos", dataset_infos.size());
    return dataset_infos;
}


std::vector<std::filesystem::path> getAllDirectories(const std::string& path)
{
    std::vector<std::filesystem::path> directories;

    if (!std::filesystem::exists(path) || !std::filesystem::is_directory(path))
    {
        spdlog::error("The provided path is not a directory.");
        return directories;
    }

    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        if (entry.is_directory())
        {
            directories.push_back(entry.path());
        }
    }
    return directories;
}

DatasetInfo getDatasetInfo(const std::string& path)
{
    DatasetInfo dataset_info;
    if (fs::is_directory(path))
    {
        std::filesystem::path dir_path(path);
        dataset_info.name = dir_path.filename().string();
        for (const auto& entry : fs::directory_iterator(path))
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
        spdlog::error("Cannot get the dataset info, the provided path {} is not a directory", path);
    }
    return dataset_info;
}


std::vector<Annotation> loadOTBAnnotations(const std::string& filename)
{
    // Each row in the ground-truth files represents the bounding box of the target in that frame,
    // (x, y, box-width, box-height).

    std::vector<Annotation> annotations;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        spdlog::error("Could not open the annotation file: {}", filename);
        return annotations;
    }
    unsigned int frame_num = 0;

    std::string line;
    while (std::getline(file, line))
    {
        // Replace commas with spaces to handle comma-separated values
        std::replace(line.begin(), line.end(), ',', ' ');

        std::istringstream iss(line);
        int x, y, width, height;
        if (iss >> x >> y >> width >> height)
        {
            Annotation annotation;
            annotation.rect = cv::Rect2f(x, y, width, height);
            annotation.frame = frame_num++;
            annotations.push_back(annotation);
        }
    }

    file.close();
    return annotations;
}

std::vector<Annotation> loadCustomAnnotations(const std::string& filename)
{
    std::vector<Annotation> annotations;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        spdlog::error("Could not open the annotation file: {}", filename);
        return annotations;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        std::string token;

        // Parse frame number
        std::getline(ss, token, ',');
        int frame = std::stoi(token);

        // Check if the line contains annotation data
        if (std::getline(ss, token, ','))
        {
            // Parse the rest of the annotation data
            float norm_x = std::stof(token);
            std::getline(ss, token, ',');
            float norm_y = std::stof(token);
            std::getline(ss, token, ',');
            float norm_width = std::stof(token);
            std::getline(ss, token, ',');
            float norm_height = std::stof(token);
            std::getline(ss, token, ',');
            int occluded = std::stoi(token);

            // Create the floating-point rectangle (cv::Rect2f)
            cv::Rect2f rect(norm_x - norm_width / 2, norm_y - norm_height / 2, norm_width, norm_height);

            Annotation annotation;
            annotation.rect = rect;
            annotation.frame = frame;
            annotation.occluded = occluded;

            annotations.push_back(annotation);
        }
        else
        {
            // Only frame number, no annotations
            Annotation annotation;
            annotation.frame = frame;
            annotations.push_back(annotation);
        }
    }

    file.close();
    return annotations;
}