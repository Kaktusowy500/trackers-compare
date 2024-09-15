#pragma once
#include <opencv2/opencv.hpp>

namespace cv {

    class TrackerModVIT : public Tracker
    {
    public:
        struct Params {
            std::string net;
            Scalar meanvalue;
            Scalar stdvalue;
            int backend;
            int target;

            Params();
        };

        TrackerModVIT();
        virtual ~TrackerModVIT();

        static Ptr<TrackerModVIT> create(const Params& parameters = Params());
        virtual float getTrackingScore() = 0;

    protected:
        virtual void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE = 0;
        virtual bool update(InputArray image, Rect& boundingBox) CV_OVERRIDE = 0;
    };

}
