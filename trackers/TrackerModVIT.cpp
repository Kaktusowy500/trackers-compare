#include "TrackerModVIT.hpp"

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

        Rect rectLast;
        float trackingScore;

        TrackerModVIT::Params params;

    protected:
        void preprocess(const Mat& src, Mat& dst, Size size);

        const Size searchSize{ 256, 256 };
        const Size templateSize{ 128, 128 };

        Mat hanningWindow;

        dnn::Net net;
        Mat image;
    };

    static void crop_image(const Mat& src, Mat& dst, Rect box, int factor)
    {
        int x = box.x, y = box.y, w = box.width, h = box.height;
        int cropSz = cvCeil(sqrt(w * h) * factor);

        int x1 = x + (w - cropSz) / 2;
        int x2 = x1 + cropSz;
        int y1 = y + (h - cropSz) / 2;
        int y2 = y1 + cropSz;

        int x1_pad = std::max(0, -x1);
        int y1_pad = std::max(0, -y1);
        int x2_pad = std::max(x2 - src.size[1] + 1, 0);
        int y2_pad = std::max(y2 - src.size[0] + 1, 0);

        Rect roi(x1 + x1_pad, y1 + y1_pad, x2 - x2_pad - x1 - x1_pad, y2 - y2_pad - y1 - y1_pad);
        Mat im_crop = src(roi);
        copyMakeBorder(im_crop, dst, y1_pad, y2_pad, x1_pad, x2_pad, BORDER_CONSTANT);
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
        int intersectionArea = (bb1 & bb2).area();
        int unionArea = bb1.area() + bb2.area() - intersectionArea;
        return static_cast<double>(intersectionArea) / unionArea;
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

    static Rect returnfromcrop(float x, float y, float w, float h, Rect resLast)
    {
        int cropWindowWH = 4 * cvFloor(sqrt(resLast.width * resLast.height));
        int x0 = resLast.x + (resLast.width - cropWindowWH) / 2;
        int y0 = resLast.y + (resLast.height - cropWindowWH) / 2;
        Rect finalRes;
        finalRes.x = cvFloor(x * cropWindowWH + x0);
        finalRes.y = cvFloor(y * cropWindowWH + y0);
        finalRes.width = cvFloor(w * cropWindowWH);
        finalRes.height = cvFloor(h * cropWindowWH);
        return finalRes;
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
        rectLast = boundingBox_;
    }

    bool TrackerModVITImpl::update(InputArray image_, Rect& boundingBoxRes)
    {
        image = image_.getMat().clone();
        Mat crop;
        crop_image(image, crop, rectLast, 4);
        Mat blob;
        preprocess(crop, blob, searchSize);
        net.setInput(blob, "search");
        std::vector<String> outputName = { "output1", "output2", "output3" };
        std::vector<Mat> outs;
        net.forward(outs, outputName);
        CV_Assert(outs.size() == 3);

        // unpack the network output
        Mat confMap = outs[0].reshape(0, { 16, 16 });
        Mat sizeMap = outs[1].reshape(0, { 2, 16, 16 });
        Mat offsetMap = outs[2].reshape(0, { 2, 16, 16 });

        multiply(confMap, (1.0 - hanningWindow), confMap);

        std::vector<Rect> maxRects;
        std::vector<double> maxScores;
        Mat confMapCopy = confMap.clone();

        //Take 5 highest scores
        for (int i = 0; i < 5; i++)
        {
            double maxVal;
            Point maxLoc;
            minMaxLoc(confMapCopy, nullptr, &maxVal, nullptr, &maxLoc);

            trackingScore = static_cast<float>(maxVal);

            float cx = (maxLoc.x + offsetMap.at<float>(0, maxLoc.y, maxLoc.x)) / 16;
            float cy = (maxLoc.y + offsetMap.at<float>(1, maxLoc.y, maxLoc.x)) / 16;
            float w = sizeMap.at<float>(0, maxLoc.y, maxLoc.x);
            float h = sizeMap.at<float>(1, maxLoc.y, maxLoc.x);

            Rect candidateRect = returnfromcrop(cx - w / 2, cy - h / 2, w, h, rectLast);
            maxRects.push_back(candidateRect);
            maxScores.push_back(trackingScore);
            confMapCopy.at<float>(maxLoc.y, maxLoc.x) = 0;
            cv::rectangle(image, candidateRect, cv::Scalar(255, 0, 0), 2, 1);
        }

        int highestScoreIndex = 0;

        // simillar confs of the best two candidates
        if (maxScores[0] < 1.2 * maxScores[1]) {
            // postprocessed scores
            std::vector<double> candidatesScores;
            // take first 3 highest scores and calculate their overlaps with other ones
            for (int i = 0; i < 3; i++)
            {
                double candidateScore = 0;
                for (int j = 0; j < 5; j++)
                {
                    candidateScore += calculate_overlap(maxRects[i], maxRects[j]) * maxScores[j];
                }
                candidateScore *= maxScores[i];
                candidatesScores.push_back(candidateScore);
            }

            auto maxCandidateScoreIter = std::max_element(candidatesScores.begin(), candidatesScores.end());
            highestScoreIndex = std::distance(candidatesScores.begin(), maxCandidateScoreIter);

        }

        rectLast = maxRects[highestScoreIndex];
        boundingBoxRes = maxRects[highestScoreIndex];
        trackingScore = maxScores[highestScoreIndex];
        return true;
    }

    float TrackerModVITImpl::getTrackingScore()
    {
        return trackingScore;
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

