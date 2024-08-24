#include "TrackerComparator.hpp"
#include "CSRTTracker.hpp"
#include "VITTracker.hpp"
#include "DaSiamTracker.hpp"
#include "DatasetUtils.hpp"
#include "VideoFileReader.hpp"
#include "ImageSequenceReader.hpp"

void TrackerComparator::loadDataset(std::string path)
{
    dataset_info = getDatasetInfo(path);
    std::cout << dataset_info << std::endl;
    ground_truths = loadRectsFromFile(dataset_info.ground_truth_paths[0]);
    std::cout << "Ground truth vector size: " << ground_truths.size() << std::endl;
}

bool TrackerComparator::setupVideoReader()
{

    if (dataset_info.dataset_type == DatasetType::OTB)
    {
        videoReader = std::make_unique<ImageSequenceReader>(dataset_info.media_path);
    }
    else if (dataset_info.dataset_type == DatasetType::Custom)
    {
        videoReader = std::make_unique<VideoFileReader>(dataset_info.media_path);
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
        colors = std::vector<cv::Scalar>({cv::Scalar(255, 50, 150), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0)});

        for (const auto &t : trackers)
        {
            evaluators.push_back(std::make_unique<TrackerPerformanceEvaluator>(t->getName()));
        }
        return true;
    }
    catch (const std::exception &e)
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
    if (videoReader->getNextFrame(frame))
    {
        for (auto &t : trackers)
        {
            t->init(frame, ground_truths[frame_count]);
        }
        cv::rectangle(frame, ground_truths[frame_count], cv::Scalar(0, 0, 255), 2, 1);
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

    while (!videoReader->isDone())
    {
        if (videoReader->getNextFrame(frame))
        {
            cv::rectangle(frame, ground_truths[frame_count], cv::Scalar(0, 0, 255), 2, 1);

            for (int i = 0; i < trackers.size(); i++)
            {
                cv::Rect bbox;
                auto start_time = std::chrono::high_resolution_clock::now();
                trackers[i]->update(frame, bbox);
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> processing_time = end_time - start_time;
                evaluators[i]->addFrameResult(ground_truths[frame_count], bbox, processing_time.count());

                bool trackingValid = (trackers[i]->getState() == TrackerState::Tracking);
                // if (trackingValid)
                // {

                auto color = trackingValid ? colors[i] : cv::Scalar(0, 0, 255);
                cv::putText(frame, trackers[i]->getName(), cv::Point(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color,
                            2);
                cv::rectangle(frame, bbox, color, 2, 1);
                // }
                std::string state_str = stateToString(trackers[i]->getState());
                cv::Scalar state_color = trackingValid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                cv::putText(frame,
                            trackers[i]->getName() + " : " + state_str + " " + std::to_string(trackers[i]->getTrackingScore()),
                            cv::Point(10, (frame.rows - 20) - 30 * i), cv::FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2);
            }

            cv::imshow("Frame", frame);
            if (cv::waitKey(30) >= 0)
                break; // Press any key to exit
            frame_count++;
        }
    }
}

void TrackerComparator::saveResults(std::string path)
{
    for (int i = 0; i < trackers.size(); i++)
    {
        std::string filename = path + "/" + trackers[i]->getName() + "_results.csv";
        evaluators[i]->saveResultsToFile(filename);
    }
    std::cout << "Results saved to: " << path << std::endl;
}
