#include <iostream>
#include <memory>
#include <random>
#include <filesystem>
#include "DatasetUtils.hpp"

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << "clip directory" << std::endl;
        return -1;
    }
    DatasetInfo dataset_info = getDatasetInfo(argv[1]);
    std::cout << dataset_info << std::endl;
    return 0;
}
