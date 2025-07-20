#pragma once

#include <chrono>

class Timer {
public:
    using time_t = std::chrono::time_point<std::chrono::high_resolution_clock>;

    Timer() {}

    void On();
    void Update();

    float GetFPS() const { return mFPS; }
    float GetElapsedTime() const { return mElapsedTime; }

    bool IsFPSUpdated() const { return mFPSUpdated; }
    void SetFPSUpdated(bool updated) { mFPSUpdated = updated; }

private:
    time_t mStartTime;
    time_t mPrevTime;
    int mFrameCount = 0;
    float mFPS = 0.0f;
    float mElapsedTime;
    bool mFPSUpdated = false;

};