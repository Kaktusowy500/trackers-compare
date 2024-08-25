#include <sstream>
#include <fstream>
#include "TrackerComparator.hpp"
#include "CSRTTracker.hpp"
#include "VITTracker.hpp"
#include "DaSiamTracker.hpp"
#include "DatasetUtils.hpp"
#include "VideoFileReader.hpp"
#include "ImageSequenceReader.hpp"

// Compare strategies
// Reset imidiately after loss, count resets and avg tracking time
// Reset after some time to allow recovery
// One init and do not count when aim was lost

// 2 validity info, one from tracker itself, one from evaluator

enum class ReinitStrategy
{
    Immediate,
    Delayed,
    OneInit
};

void TrackerComparator::loadDataset(std::string path)
{
    dataset_info = getDatasetInfo(path);
    std::cout << dataset_info << std::endl;
    if (dataset_info.dataset_type == DatasetType::Custom)
    {
        ground_truths = loadCustomAnnotations(dataset_info.ground_truth_paths[0]);
    }
    else if (dataset_info.dataset_type == DatasetType::OTB)
    {
        ground_truths = loadOTBAnnotations(dataset_info.ground_truth_paths[0]);
    }
    std::cout << "Ground truth vector size: " << ground_truths.size() << std::endl;
}

bool TrackerComparator::setupVideoReader()
{

    if (dataset_info.dataset_type == DatasetType::OTB)
    {
        video_reader = std::make_unique<ImageSequenceReader>(dataset_info.media_path);
    }
    else if (dataset_info.dataset_type == DatasetType::Custom)
    {
        video_reader = std::make_unique<VideoFileReader>(dataset_info.media_path);
    }
    else
    {
        std::cout << "Unknown dataset type." << std::endl;
        return false;
    }
    return true;
}

bool TrackerComparator::setupTrackersAndEvaluators()
{
    try
    {
        trackers.push_back(std::make_unique<CSRTTracker>());
        trackers.push_back(std::make_unique<VITTracker>());
        trackers.push_back(std::make_unique<DaSiamTracker>());
        colors = std::vector<cv::Scalar>({ cv::Scalar(255, 50, 150), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0) });

        for (const auto& t : trackers)
        {
            evaluators.push_back(std::make_unique<TrackerPerformanceEvaluator>(t->getName()));
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
        std::cout << "Error reading first frame." << std::endl;
        return false;
    }
    return true;
}

void TrackerComparator::runEvaluation()
{
    if (!readFirstFrameAndInit())
        return;

    while (!video_reader->isDone())
    {
        if (video_reader->getNextFrame(frame))
        {
            cv::rectangle(frame, ground_truths[frame_count].rect, cv::Scalar(0, 0, 255), 2, 1);

            for (int i = 0; i < trackers.size(); i++)
            {
                cv::Rect bbox;
                auto start_time = std::chrono::high_resolution_clock::now();
                trackers[i]->update(frame, bbox);
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> processing_time = end_time - start_time;
                bool is_valid_by_evaluator = evaluators[i]->addFrameResult(ground_truths[frame_count].rect, bbox, processing_time.count());

                bool tracking_valid = (trackers[i]->getState() == TrackerState::Tracking);
                if (!is_valid_by_evaluator)
                {
                    trackers[i]->init(frame, ground_truths[frame_count].rect);
                    evaluators[i]->trackingReinited();
                }
                // if (tracking_valid)
                // {

                auto color = tracking_valid ? colors[i] : cv::Scalar(0, 0, 255);
                cv::putText(frame, trackers[i]->getName(), cv::Point(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color,
                    2);
                cv::rectangle(frame, bbox, color, 2, 1);
                // }
                std::string state_str = stateToString(trackers[i]->getState());
                cv::Scalar state_color = tracking_valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                cv::putText(frame,
                    trackers[i]->getName() + " : " + state_str + " " + std::to_string(trackers[i]->getTrackingScore()),
                    cv::Point(10, (frame.rows - 20) - 30 * i), cv::FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2);
            }

            cv::imshow("Frame", frame);
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
        std::cout << "Tracker not found." << std::endl;
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

    std::cout << "Results saved to: " << path << std::endl;
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