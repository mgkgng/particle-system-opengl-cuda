#pragma once

#include <chrono>

class Timer {
public:
    using time_t = std::chrono::time_point<std::chrono::high_resolution_clock>;

    Timer() = default;

    void On();
    void Update();
    void Reset() { mStartTime = std::chrono::high_resolution_clock::now(); }

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
    bool mIsOn = false;
};