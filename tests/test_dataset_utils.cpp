#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "DatasetUtils.hpp"

namespace fs = std::filesystem;

class DatasetInfoTest : public ::testing::Test {
protected:
    fs::path testDir;

    void SetUp() override {
        // Create a temporary directory for testing
        testDir = fs::temp_directory_path() / "test_data";
        fs::create_directory(testDir);

        // Set up directory structure
        fs::create_directory(testDir / "dataset1");
        std::ofstream(testDir / "dataset1/video.mp4").close();
        std::ofstream(testDir / "dataset1/truth1.txt").close();

        fs::create_directory(testDir / "dataset2");
        fs::create_directory(testDir / "dataset2/img");
        std::ofstream(testDir / "dataset2/truth1.txt").close();

        fs::create_directory(testDir / "dataset3");
        std::ofstream(testDir / "dataset3/video.mp4").close();

    }

    void TearDown() override {
        // Clean up temporary directory after tests
        fs::remove_all(testDir);
    }
};

TEST_F(DatasetInfoTest, HandlesMp4AndTxtFiles) {
    DatasetInfo info = getDatasetInfo((testDir / "dataset1").string());

    EXPECT_EQ(info.media_path, (testDir / "dataset1/video.mp4").string());
    EXPECT_EQ(info.dataset_type, DatasetType::Custom);
    ASSERT_EQ(info.ground_truth_paths.size(), 1);
    EXPECT_EQ(info.ground_truth_paths[0], (testDir / "dataset1/truth1.txt").string());
}

TEST_F(DatasetInfoTest, HandlesImgAndTxtFiles) {
    DatasetInfo info = getDatasetInfo((testDir / "dataset2").string());

    EXPECT_EQ(info.media_path, (testDir / "dataset2/img").string());
    EXPECT_EQ(info.dataset_type, DatasetType::OTB);
    ASSERT_EQ(info.ground_truth_paths.size(), 1);
    EXPECT_EQ(info.ground_truth_paths[0], (testDir / "dataset2/truth1.txt").string());
}

TEST_F(DatasetInfoTest, HandlesMp4WithoutTxt) {
    DatasetInfo info = getDatasetInfo((testDir / "dataset3").string());

    EXPECT_EQ(info.media_path, (testDir / "dataset3/video.mp4").string());
    EXPECT_EQ(info.dataset_type, DatasetType::Custom);
    EXPECT_TRUE(info.ground_truth_paths.empty());
}

TEST_F(DatasetInfoTest, HandlesNonDirectoryPath) {
    DatasetInfo info = getDatasetInfo((testDir / "non_existing_directory").string());

    EXPECT_EQ(info.media_path, "");
    EXPECT_TRUE(info.ground_truth_paths.empty());
}

