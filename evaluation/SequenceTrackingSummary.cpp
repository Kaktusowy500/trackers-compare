#include "SequenceTrackingSummary.hpp"

YAML::Emitter& operator<<(YAML::Emitter& out, const SequenceTrackingSummary& summary)
{
    out << YAML::BeginMap;
    out << YAML::Key << "avg_overlap" << YAML::Value << summary.avg_overlap;
    out << YAML::Key << "avg_overlap_std" << YAML::Value << summary.avg_overlap_std;
    out << YAML::Key << "avg_cle" << YAML::Value << summary.avg_cle;
    out << YAML::Key << "avg_cle_std" << YAML::Value << summary.avg_cle_std;
    out << YAML::Key << "avg_time" << YAML::Value << summary.avg_time;
    out << YAML::Key << "avg_time_std" << YAML::Value << summary.avg_time_std;
    out << YAML::Key << "success_rt" << YAML::Value << summary.success_rt;
    out << YAML::Key << "reinit_cnt" << YAML::Value << summary.reinit_cnt;
    out << YAML::EndMap;
    return out;
}