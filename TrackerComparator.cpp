#include <sstream>
#include <fstream>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h> 
#include "TrackerComparator.hpp"
#include "CSRTTracker.hpp"
#include "VITTracker.hpp"
#include "DaSiamTracker.hpp"
#include "DatasetUtils.hpp"
#include "VideoFileReader.hpp"
#include "ImageSequenceReader.hpp"


TrackerComparator::TrackerComparator(const YAML::Node& config) : config(config)
{
    parseReinitStrategy(config["reinit_strategy"].as<std::string>());
}

void TrackerComparator::parseReinitStrategy(const std::string& strategy)
{
    if (strategy == "immediate")
        reinit_strategy = ReinitStrategy::Immediate;
    else if (strategy == "delayed")
        reinit_strategy = ReinitStrategy::Delayed;
    else if (strategy == "one_init")
        reinit_strategy = ReinitStrategy::OneInit;
    else
        spdlog::error("Unknown reinit strategy: {}", strategy);
}

void TrackerComparator::loadDataset(std::string path, bool only_video)
{
    if (only_video)
    {
        dataset_info.media_path = path;
        dataset_info.dataset_type = DatasetType::VideoOnly;
        return;
    }

    dataset_info = getDatasetInfo(path);
    spdlog::debug("Dataset info: \n{}", fmt::streamed(dataset_info));
    if (dataset_info.dataset_type == DatasetType::Custom)
    {
        ground_truths = loadCustomAnnotations(dataset_info.ground_truth_paths[0]);
    }
    else if (dataset_info.dataset_type == DatasetType::OTB)
    {
        ground_truths = loadOTBAnnotations(dataset_info.ground_truth_paths[0]);
    }
    spdlog::debug("Ground truth vector size: {}", ground_truths.size());
}

bool TrackerComparator::setupVideoReader()
{

    if (dataset_info.dataset_type == DatasetType::OTB)
    {
        video_reader = std::make_unique<ImageSequenceReader>(dataset_info.media_path);
    }
    else if (dataset_info.dataset_type == DatasetType::Custom || dataset_info.dataset_type == DatasetType::VideoOnly)
    {
        video_reader = std::make_unique<VideoFileReader>(dataset_info.media_path);
    }
    else
    {
        spdlog::error("Unknown dataset type");
        return false;
    }
    return true;
}

bool TrackerComparator::setupTrackersAndEvaluators()
{
    try
    {
        trackers.push_back(std::make_unique<CSRTTracker>());
        trackers.push_back(std::make_unique<VITTracker>(config["trackers"]["vit"]["score_thresh"].as<double>()));
        trackers.push_back(std::make_unique<DaSiamTracker>(config["trackers"]["dasiam"]["score_thresh"].as<double>()));
        colors = std::vector<cv::Scalar>({ cv::Scalar(255, 50, 150), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0) });

        TrackerPerformanceEvaluatorArgs args;
        args.overlap_thresh = config["evaluation"]["overlap_thresh"].as<double>();
        args.center_error_thresh = config["evaluation"]["center_error_thresh"].as<double>();
        for (const auto& t : trackers)
        {
            args.tracker_name = t->getName();
            evaluators.push_back(std::make_unique<TrackerPerformanceEvaluator>(args));
        }
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return false;
    }
}

bool TrackerComparator::setupComponents()
{
    if (setupVideoReader() && setupTrackersAndEvaluators())
    {
        return true;
    }
    return false;
}

bool TrackerComparator::readFirstFrameAndInit()
{
    if (video_reader->getNextFrame(frame))
    {
        if (dataset_info.dataset_type == DatasetType::Custom)
        {
            convertGTToNonNormalized(frame.cols, frame.rows);
        }
        for (auto& t : trackers)
        {
            t->init(frame, ground_truths[frame_count].rect);
        }
        cv::rectangle(frame, ground_truths[frame_count].rect, cv::Scalar(0, 0, 255), 2, 1);
        cv::imshow("First frame", frame);
        frame_count++;
    }
    else
    {
        spdlog::error("Error reading first frame");
        return false;
    }
    return true;
}

void TrackerComparator::applyReinitStrategy(const cv::Mat& frame, int index, ValidationStatus reason)
{

    if (reinit_strategy == ReinitStrategy::Immediate)
    {
        if (ground_truths[frame_count].occluded != 1)
        {
            spdlog::debug("Try to apply reninit strategy to tracker {}, reason {}", trackers[index]->getName(), ValidationStatusToString(reason));
            trackers[index]->init(frame, ground_truths[frame_count].rect);
            evaluators[index]->trackingReinited();
        }
    }
    else if (reinit_strategy == ReinitStrategy::OneInit)
    {
        trackers[index]->setState(TrackerState::Lost);
    }

    // TODO implement delayed reinit strategy
}

void TrackerComparator::runEvaluation()
{
    if (!readFirstFrameAndInit())
        return;

    while (!video_reader->isDone())
    {
        if (video_reader->getNextFrame(frame))
        {
            cv::Mat frame_vis = frame.clone();
            if (frame_count >= ground_truths.size())
            {
                spdlog::error("Ground truth vector size exceeded");
                break;
            }
            cv::rectangle(frame_vis, ground_truths[frame_count].rect, cv::Scalar(0, 0, 255), 2, 1);

            for (int i = 0; i < trackers.size(); i++)
            {
                cv::Rect bbox;

                auto start_time = std::chrono::high_resolution_clock::now();
                if (trackers[i]->getState() != TrackerState::Lost && trackers[i]->getState() != TrackerState::ToBeReinited)
                    trackers[i]->update(frame, bbox);
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> processing_time = end_time - start_time;
                ValidationStatus valid_status = evaluators[i]->validateAndAddResult(ground_truths[frame_count].rect, bbox, processing_time.count(), trackers[i]->getState() == TrackerState::Lost);

                bool tracking_valid = (trackers[i]->getState() == TrackerState::Tracking);
                if (valid_status != ValidationStatus::Valid && trackers[i]->getState() != TrackerState::Lost)
                {
                    tracking_valid = false;
                    applyReinitStrategy(frame, i, valid_status);
                }
                // if (tracking_valid)
                // {

                auto color = tracking_valid ? colors[i] : cv::Scalar(0, 0, 255);
                cv::putText(frame_vis, trackers[i]->getName(), cv::Point(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color,
                    2);
                cv::rectangle(frame_vis, bbox, color, 2, 1);
                // }
                std::string state_str = stateToString(trackers[i]->getState());
                cv::Scalar state_color = tracking_valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                cv::putText(frame_vis,
                    trackers[i]->getName() + " : " + state_str + " " + std::to_string(trackers[i]->getTrackingScore()),
                    cv::Point(10, (frame_vis.rows - 20) - 30 * i), cv::FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2);
            }

            cv::imshow("Frame", frame_vis);
            if (cv::waitKey(10) >= 0)
                break; // Press any key to exit
            frame_count++;
        }
    }
}

void TrackerComparator::runPreview(std::string tracker_name)
{
    int tracker_id = -1;
    for (int i = 0; i < trackers.size(); i++)
    {
        if (trackers[i]->getName() == tracker_name)
        {
            tracker_id = i;
            break;
        }
    }
    if (tracker_id < -1)
    {
        spdlog::error("Tracker not found");
        return;
    }


    while (!video_reader->isDone())
    {

        if (video_reader->getNextFrame(frame))
        {
            cv::Rect bbox;
            bool tracking_valid = (trackers[tracker_id]->getState() == TrackerState::Tracking);
            if (tracking_valid)
            {
                auto start_time = std::chrono::high_resolution_clock::now();
                trackers[tracker_id]->update(frame, bbox);
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> processing_time = end_time - start_time; // print is somewhere

                auto color = tracking_valid ? colors[tracker_id] : cv::Scalar(0, 0, 255);
                cv::putText(frame, trackers[tracker_id]->getName(), cv::Point(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color,
                    2);
                cv::rectangle(frame, bbox, color, 2, 1);
                cv::putText(frame,
                    "Processing time: " + std::to_string(processing_time.count()),
                    cv::Point(10, (frame.rows - 50)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
            }
            std::string state_str = stateToString(trackers[tracker_id]->getState());
            cv::Scalar state_color = tracking_valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::putText(frame,
                trackers[tracker_id]->getName() + " : " + state_str + " " + std::to_string(trackers[tracker_id]->getTrackingScore()),
                cv::Point(10, (frame.rows - 20)), cv::FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2);

            cv::imshow("Frame", frame);
            auto key = cv::waitKey(30);
            if (key == 's')
            {
                cv::Rect bbox = cv::selectROI("Frame", frame, false);
                trackers[tracker_id]->init(frame, bbox);
            }
            else if (key == 'q')
                break; // Press any key to exit
        }
    }

}


void TrackerComparator::saveResults(std::string path)
{

    std::string summary_file_path = path + "/" + "summary.csv";
    std::ofstream summary_file(summary_file_path);
    if (!summary_file.is_open())
    {
        std::cerr << "Could not open the file: " << summary_file_path << std::endl;
        return;
    }
    summary_file << "Tracker Name,Reinit" << std::endl;

    for (int i = 0; i < trackers.size(); i++)
    {
        std::string filename = path + "/" + trackers[i]->getName() + "_results.csv";
        evaluators[i]->saveResultsToFile(filename);
        summary_file << trackers[i]->getName() << "," << evaluators[i]->getReinitCount() << std::endl;
    }

    spdlog::info("Results saved to: {}", path);
}

void TrackerComparator::convertGTToNonNormalized(int imgWidth, int imgHeight)
{
    for (auto& gt : ground_truths)
    {
        float x = gt.rect.x * imgWidth;
        float y = gt.rect.y * imgHeight;
        float width = gt.rect.width * imgWidth;
        float height = gt.rect.height * imgHeight;
        gt.rect = cv::Rect2f(x, y, width, height);
    }
}