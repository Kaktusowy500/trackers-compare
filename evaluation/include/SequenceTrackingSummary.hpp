#pragma once

#include <yaml-cpp/yaml.h>

struct SequenceTrackingSummary
{
    double avg_overlap;
    double avg_overlap_std;
    double avg_cle;
    double avg_cle_std;
    double avg_time; 
    double avg_time_std; 
    double success_rt;
    unsigned int reinit_cnt;
};

YAML::Emitter& operator<<(YAML::Emitter& out, const SequenceTrackingSummary& summary);
