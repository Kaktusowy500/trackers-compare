#include "ITracker.hpp"
#include <spdlog/spdlog.h>

std::string stateToString(TrackerState state)
{
    static std::map<TrackerState, std::string> map = {
        {TrackerState::Ready, "Ready"},
        {TrackerState::Tracking, "Tracking"},
        {TrackerState::Recovering, "Recovering"},
        {TrackerState::Lost, "Lost"},
        {TrackerState::ToBeReinited, "ToBeReinited"} };
    return map[state];
}

double ITracker::getTrackingScore()
{
    return -1;
}
std::string ITracker::getName()
{
    return name;
}

TrackerState ITracker::getState()
{
    return state;
}

ITracker::ITracker()
{
    state = TrackerState::Ready;
}

void ITracker::setState(TrackerState s)
{
    if (state == s)
        return;
    spdlog::info("Tracker: {} state changed from {} to {}", name, stateToString(state), stateToString(s));
    state = s;
}