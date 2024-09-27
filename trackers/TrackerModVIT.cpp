#include "TrackerModVIT.hpp"
#include <limits> 

namespace cv {

    TrackerModVIT::TrackerModVIT()
    {
    }

    TrackerModVIT::~TrackerModVIT()
    {
    }

    TrackerModVIT::Params::Params()
    {
        net = "vitTracker.onnx";
        meanvalue = Scalar{ 0.485, 0.456, 0.406 };
        stdvalue = Scalar{ 0.229, 0.224, 0.225 };
#ifdef HAVE_OPENCV_DNN
        backend = dnn::DNN_BACKEND_DEFAULT;
        target = dnn::DNN_TARGET_CPU;
#else
        backend = -1;  // invalid value
        target = -1;  // invalid value
#endif
    }

#ifdef HAVE_OPENCV_DNN

    class TrackerModVITImpl : public TrackerModVIT
    {
    public:
        TrackerModVITImpl(const TrackerModVIT::Params& parameters)
            : params(parameters)
        {
            net = dnn::readNet(params.net);
            CV_Assert(!net.empty());

            net.setPreferableBackend(params.backend);
            net.setPreferableTarget(params.target);
        }

        void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
        bool update(InputArray image, Rect& boundingBox) CV_OVERRIDE;
        float getTrackingScore() CV_OVERRIDE;

        Rect rect_last;
        float tracking_score;

        TrackerModVIT::Params params;

    protected:
        void preprocess(const Mat& src, Mat& dst, Size size);
        double compare_with_template(const Mat& img);
        void update_template(const Mat& img);
        void fix_rect(Rect& rect, const Mat& image);

        const Size searchSize{ 256, 256 };
        const Size templateSize{ 128, 128 };

        Mat hanningWindow;

        dnn::Net net;
        Mat image;
        cv::Mat template_descriptors;
    };

    static void crop_image(const Mat& src, Mat& dst, Rect box, int factor)
    {
        int x = box.x, y = box.y, w = box.width, h = box.height;
        int crop_sz = cvCeil(sqrt(w * h) * factor);

        int x1 = x + (w - crop_sz) / 2;
        int x2 = x1 + crop_sz;
        int y1 = y + (h - crop_sz) / 2;
        int y2 = y1 + crop_sz;

        int x1_pad = std::max(0, -x1);
        int y1_pad = std::max(0, -y1);
        int x2_pad = std::max(x2 - src.size[1] + 1, 0);
        int y2_pad = std::max(y2 - src.size[0] + 1, 0);

        Rect roi(x1 + x1_pad, y1 + y1_pad, x2 - x2_pad - x1 - x1_pad, y2 - y2_pad - y1 - y1_pad);
        Mat im_crop = src(roi);
        copyMakeBorder(im_crop, dst, y1_pad, y2_pad, x1_pad, x2_pad, BORDER_CONSTANT);
    }

    double TrackerModVITImpl::compare_with_template(const cv::Mat& img)
    {
        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat img_decscriptors;

        orb->detectAndCompute(img, cv::Mat(), keypoints, img_decscriptors);

        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<cv::DMatch> matches;
        if (img_decscriptors.empty())
        {
            return -1;
        }
        matcher.match(template_descriptors, img_decscriptors, matches);
        // std::cout << "matches size: " << matches.size() << "img desc size: " << img_decscriptors.rows << " template desc size: " << template_descriptors.rows << std::endl;
        std::sort(matches.begin(), matches.end());
        // for (auto & m : matches)
        // {
        //     std::cout << m.distance  << " ";
        // }
        // std::cout << std::endl;

        double sumDistances = 0.0;
        int count = std::min(10, static_cast<int>(matches.size()));
        for (int i = 0; i < count; ++i) {
            sumDistances += matches[i].distance;
        }
        double averageDistance = count > 0 ? sumDistances / count : -1;
        return averageDistance;
    }
    void TrackerModVITImpl::fix_rect(Rect& rect, const Mat& image)
    {
        rect.x = rect.x < 0 ? 0 : rect.x;
        rect.y = rect.y < 0 ? 0 : rect.y;
        rect.width = (rect.width + rect.x) > image.cols ? image.cols - rect.x : rect.width;
        rect.height = (rect.height + rect.y) > image.rows ? image.rows - rect.y : rect.height;
    }


    void TrackerModVITImpl::update_template(const cv::Mat& img)
    {
        std::cout << "update template" << std::endl;
        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        orb->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
        if (descriptors.rows > 10)
        {
            template_descriptors = descriptors;
        }
    }

    void TrackerModVITImpl::preprocess(const Mat& src, Mat& dst, Size size)
    {
        Mat mean = Mat(size, CV_32FC3, params.meanvalue);
        Mat std = Mat(size, CV_32FC3, params.stdvalue);
        mean = dnn::blobFromImage(mean, 1.0, Size(), Scalar(), false);
        std = dnn::blobFromImage(std, 1.0, Size(), Scalar(), false);

        Mat img;
        resize(src, img, size);

        dst = dnn::blobFromImage(img, 1.0, Size(), Scalar(), false);
        dst /= 255;
        dst = (dst - mean) / std;
    }

    double calculate_overlap(const cv::Rect& bb1, const cv::Rect& bb2)
    {
        int intersection_area = (bb1 & bb2).area();
        int union_area = bb1.area() + bb2.area() - intersection_area;
        return static_cast<double>(intersection_area) / union_area;
    }

    void postprocessDistances(std::vector<double>& distances) {
        double maxDist = *std::max_element(distances.begin(), distances.end());
        for (auto& dist : distances)
        {
            if (dist < 0)
                dist = maxDist;
        }
    }

    std::vector<double> normalizeDistances(const std::vector<double>& distances) {
        double maxDist = *std::max_element(distances.begin(), distances.end());
        double minDist = *std::min_element(distances.begin(), distances.end());

        if (maxDist - minDist < 0.0001)
            return std::vector<double>(distances.size(), 0.0);

        std::vector<double> normalizedDists(distances.size());
        for (size_t i = 0; i < distances.size(); ++i) {
            normalizedDists[i] = (distances[i] - minDist) / (maxDist - minDist);
        }

        return normalizedDists;
    }

    static Mat hann1d(int sz, bool centered = true) {
        Mat hanningWindow(sz, 1, CV_32FC1);
        float* data = hanningWindow.ptr<float>(0);

        if (centered) {
            for (int i = 0; i < sz; i++) {
                float val = 0.5f * (1.f - std::cos(static_cast<float>(2 * M_PI / (sz + 1)) * (i + 1)));
                data[i] = val;
            }
        }
        else {
            int half_sz = sz / 2;
            for (int i = 0; i <= half_sz; i++) {
                float val = 0.5f * (1.f + std::cos(static_cast<float>(2 * M_PI / (sz + 2)) * i));
                data[i] = val;
                data[sz - 1 - i] = val;
            }
        }

        return hanningWindow;
    }

    static Mat hann2d(Size size, bool centered = true) {
        int rows = size.height;
        int cols = size.width;

        Mat hanningWindowRows = hann1d(rows, centered);
        Mat hanningWindowCols = hann1d(cols, centered);

        Mat hanningWindow = hanningWindowRows * hanningWindowCols.t();

        return hanningWindow;
    }

    static Rect returnfromcrop(float x, float y, float w, float h, Rect res_Last)
    {
        int cropwindowwh = 4 * cvFloor(sqrt(res_Last.width * res_Last.height));
        int x0 = res_Last.x + (res_Last.width - cropwindowwh) / 2;
        int y0 = res_Last.y + (res_Last.height - cropwindowwh) / 2;
        Rect finalres;
        finalres.x = cvFloor(x * cropwindowwh + x0);
        finalres.y = cvFloor(y * cropwindowwh + y0);
        finalres.width = cvFloor(w * cropwindowwh);
        finalres.height = cvFloor(h * cropwindowwh);
        return finalres;
    }

    void TrackerModVITImpl::init(InputArray image_, const Rect& boundingBox_)
    {
        image = image_.getMat().clone();
        Mat crop;
        crop_image(image, crop, boundingBox_, 2);
        Mat blob;
        preprocess(crop, blob, templateSize);
        net.setInput(blob, "template");
        Size size(16, 16);
        hanningWindow = hann2d(size, false);
        rect_last = boundingBox_;
        update_template(image(rect_last));
    }

    bool TrackerModVITImpl::update(InputArray image_, Rect& boundingBoxRes)
    {
        image = image_.getMat().clone();
        Mat crop;
        crop_image(image, crop, rect_last, 4);
        Mat blob;
        preprocess(crop, blob, searchSize);
        net.setInput(blob, "search");
        std::vector<String> outputName = { "output1", "output2", "output3" };
        std::vector<Mat> outs;
        net.forward(outs, outputName);
        CV_Assert(outs.size() == 3);

        Mat conf_map = outs[0].reshape(0, { 16, 16 });
        Mat size_map = outs[1].reshape(0, { 2, 16, 16 });
        Mat offset_map = outs[2].reshape(0, { 2, 16, 16 });

        multiply(conf_map, (1.0 - hanningWindow), conf_map);

        std::vector<Rect> maxRects;
        std::vector<double> maxScores;
        std::vector<double> avgSimilarityDistances;
        Mat conf_map_copy = conf_map.clone();

        static constexpr int numCandidates = 5;

        //Take 5 highest scores
        for (int i = 0; i < numCandidates; i++)
        {
            double maxVal;
            Point maxLoc;
            minMaxLoc(conf_map_copy, nullptr, &maxVal, nullptr, &maxLoc);

            tracking_score = static_cast<float>(maxVal);

            float cx = (maxLoc.x + offset_map.at<float>(0, maxLoc.y, maxLoc.x)) / 16;
            float cy = (maxLoc.y + offset_map.at<float>(1, maxLoc.y, maxLoc.x)) / 16;
            float w = size_map.at<float>(0, maxLoc.y, maxLoc.x);
            float h = size_map.at<float>(1, maxLoc.y, maxLoc.x);

            Rect candidate_rect = returnfromcrop(cx - w / 2, cy - h / 2, w, h, rect_last);
            fix_rect(candidate_rect, image);
            // std::cout << candidate_rect << "score: " << tracking_score << std::endl;
            maxRects.push_back(candidate_rect);
            maxScores.push_back(tracking_score);
            avgSimilarityDistances.push_back(compare_with_template(image(candidate_rect)));
            conf_map_copy.at<float>(maxLoc.y, maxLoc.x) = 0;
            cv::rectangle(image, candidate_rect, cv::Scalar(255, 0, 0), 2, 1);
        }
        cv::imshow("ModVIT", image);
        postprocessDistances(avgSimilarityDistances);
        std::vector<double> normalizedDists = normalizeDistances(avgSimilarityDistances);

        int highestScoreIndex = 0;
        // simillar confs of the best two candidates
        if (maxScores[0] < 1.2 * maxScores[1]) {
            // postprocessed scores
            std::vector<double> candidatesScores;
            // take first 3 highest scores and calculate their overlaps with other ones
            for (int i = 0; i < 3; i++)
            {
                double candidateScore = 0;
                for (int j = 0; j < numCandidates; j++)
                {
                    candidateScore += calculate_overlap(maxRects[i], maxRects[j]) * maxScores[j];
                }
                candidateScore *= maxScores[i];
                double featureDistanceFactor = normalizedDists[i] * 0.05;
                std::cout << "candidate score: " << candidateScore << "normalizedDist" << featureDistanceFactor << std::endl;
                // candidateScore -= featureDistanceFactor;
                candidatesScores.push_back(candidateScore);
            }

            auto maxCandidateScoreIter = std::max_element(candidatesScores.begin(), candidatesScores.end());
            highestScoreIndex = std::distance(candidatesScores.begin(), maxCandidateScoreIter);
            if (highestScoreIndex != 0)
                std::cout << "non zero index"; // for debugging

        }
        // std::cout << "Score: " << compare_with_template(image(maxRects[highestScoreIndex])) << std::endl;
        if (maxScores[highestScoreIndex] > 0.7)
        {
            update_template(image(maxRects[highestScoreIndex]));
        }

        rect_last = maxRects[highestScoreIndex];
        boundingBoxRes = rect_last;
        tracking_score = maxScores[highestScoreIndex];
        return true;
    }

    float TrackerModVITImpl::getTrackingScore()
    {
        return tracking_score;
    }

    Ptr<TrackerModVIT> TrackerModVIT::create(const TrackerModVIT::Params& parameters)
    {
        return makePtr<TrackerModVITImpl>(parameters);
    }

#else  // OPENCV_HAVE_DNN
    Ptr<TrackerModVIT> TrackerModVIT::create(const TrackerModVIT::Params& parameters)
    {
        CV_UNUSED(parameters);
        CV_Error(Error::StsNotImplemented, "to use vittrack, the tracking module needs to be built with opencv_dnn !");
    }
#endif  // OPENCV_HAVE_DNN
}

