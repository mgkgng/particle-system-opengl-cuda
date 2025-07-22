#include "Timer.hpp"

void Timer::On() {
    mStartTime = std::chrono::high_resolution_clock::now(); 
    mPrevTime = mStartTime;
    mIsOn = true;
}

void Timer::Update() {
    if (!mIsOn) On();
    
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = now - mPrevTime;
    std::chrono::duration<float> totalElapsed = now - mStartTime;
    mElapsedTime = totalElapsed.count();

    if (elapsed.count() >= 1.0f) {
        mFPS = mFrameCount / elapsed.count();
        mPrevTime = now;
        mFrameCount = 0;
        mFPSUpdated = true;
    } else {
        mFrameCount++;
    }
}
