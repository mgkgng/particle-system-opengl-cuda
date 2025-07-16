#pragma once

#include <chrono>

class Timer {
public:
    using time_t = std::chrono::time_point<std::chrono::high_resolution_clock>;

    Timer(float checkInterval) : mCheckInterval(checkInterval) {}

    void On() { mPrevTime = std::chrono::high_resolution_clock::now(); }

    void Update() {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = now - mPrevTime;

        if (elapsed.count() >= mCheckInterval) {
            mFPS = mFrameCount / elapsed.count();
            mPrevTime = now;
            mFrameCount = 0;
        } else {
            mFrameCount++;
        }
    }

    float GetFPS() const { return mFPS; }

private:
    time_t mPrevTime;
    int mFrameCount = 0;
    float mFPS = 0.0f;
    float mCheckInterval = 1.0f;

};