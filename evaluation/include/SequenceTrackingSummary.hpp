#pragma once

#include <yaml-cpp/yaml.h>

struct SequenceTrackingSummary
{
    double average_overlap;
    double average_error;
    double average_processing_time;
    double valid_frame_percent;
    unsigned int reinit_count;
};

YAML::Emitter& operator<<(YAML::Emitter& out, const SequenceTrackingSummary& summary);
