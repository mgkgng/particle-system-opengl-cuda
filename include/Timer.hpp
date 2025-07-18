#pragma once

#include <chrono>

class Timer {
public:
    using time_t = std::chrono::time_point<std::chrono::high_resolution_clock>;

    Timer() {}

    void On() { mPrevTime = std::chrono::high_resolution_clock::now(); }

    void Update() {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = now - mPrevTime;

        if (elapsed.count() >= 1.0f) {
            mFPS = mFrameCount / elapsed.count();
            mPrevTime = now;
            mFrameCount = 0;
            mFPSUpdated = true;
        } else {
            mFrameCount++;
        }
    }

    float GetFPS() const { return mFPS; }

    bool IsFPSUpdated() const { return mFPSUpdated; }
    void SetFPSUpdated(bool updated) { mFPSUpdated = updated; }

private:
    time_t mPrevTime;
    int mFrameCount = 0;
    float mFPS = 0.0f;
    bool mFPSUpdated = false;

};