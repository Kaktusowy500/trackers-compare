#include "SequenceTrackingSummary.hpp"

YAML::Emitter& operator<<(YAML::Emitter& out, const SequenceTrackingSummary& summary)
{
    out << YAML::BeginMap;
    out << YAML::Key << "average_overlap" << YAML::Value << summary.average_overlap;
    out << YAML::Key << "average_error" << YAML::Value << summary.average_error;
    out << YAML::Key << "average_processing_time" << YAML::Value << summary.average_processing_time;
    out << YAML::Key << "valid_frame_percent" << YAML::Value << summary.valid_frame_percent;
    out << YAML::Key << "reinit_count" << YAML::Value << summary.reinit_count;
    out << YAML::EndMap;
    return out;
}