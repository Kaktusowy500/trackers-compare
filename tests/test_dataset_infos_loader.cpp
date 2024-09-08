#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "DatasetUtils.hpp"

namespace fs = std::filesystem;

class LoadDatasetInfosTest : public ::testing::Test {
protected:
    fs::path testDir;

    void SetUp() override {
        testDir = fs::temp_directory_path() / "test_data";
        fs::create_directory(testDir);

        // Set up directory structure
        fs::create_directory(testDir / "instance1");
        std::ofstream(testDir / "instance1/video.mp4").close();
        std::ofstream(testDir / "instance1/truth1.txt").close();

        fs::create_directory(testDir / "instance2");
        fs::create_directory(testDir / "instance2/img");
        std::ofstream(testDir / "instance2/truth1.txt").close();

        fs::create_directory(testDir / "instance3");
        std::ofstream(testDir / "instance3/video.mp4").close();
        std::ofstream(testDir / "instance3/truth1.txt").close();

    }

    void TearDown() override {
        // Clean up temporary directory after tests
        fs::remove_all(testDir);
    }
};


TEST_F(LoadDatasetInfosTest, HandleCustomDatasetWithManyInstances) {
    auto dataset_infos = loadDatasetInfos(testDir.string());

    EXPECT_EQ(dataset_infos.size(), 3);
}

TEST_F(LoadDatasetInfosTest, HandleCustomDatasetOneInstance) {
    auto dataset_infos = loadDatasetInfos(testDir.string()+"/instance1");

    ASSERT_EQ(dataset_infos.size(), 1);
    EXPECT_EQ(dataset_infos[0].name, "instance1");
    EXPECT_EQ(dataset_infos[0].media_path, testDir.string()+"/instance1/video.mp4");
}

TEST(LoadDatasetInfosTestNoFixture, HandleCustomDatasetNoInstances) {
    fs::path testDir = fs::temp_directory_path() / "test_data";
    fs::create_directory(testDir);
    auto dataset_infos = loadDatasetInfos(testDir.string());

    ASSERT_EQ(dataset_infos.size(), 0);
    fs::remove_all(testDir);
}


