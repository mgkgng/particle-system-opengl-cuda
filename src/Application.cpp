#include "Application.hpp"

Application::Application(ProgramConfig& programConfig)
    : mProgramConfig(programConfig)
    , mParticleSystem(programConfig.mParticleCount, programConfig.mShapeMode, &mTimer)
    , mRenderer(mWindow.GetWindow(), &mCamera)
    , mInputHandler(mWindow.GetWindow(), &mCamera, &mProgramConfig, &mParticleSystem, &mTimer) { 
        mWindow.SetWindowUserPointer(&mInputHandler);
    }

bool Application::InitCUDA() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA-compatible GPU found: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    std::cout << "[CUDA] " << deviceCount << " device(s) found." << std::endl;

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
    std::cout << "[CUDA] Device Name: " << deviceProp.name << std::endl;
    std::cout << "[CUDA] Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "[CUDA] Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;

    // Testing kernel launch
    int* devPtr = nullptr;
    if(cudaMalloc(&devPtr, sizeof(int)) != cudaSuccess) {
        std::cerr << "[CUDA] Memory allocation failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    cudaFree(devPtr);

    std::cout << "[CUDA] Initialization completed successfully." << std::endl;
    return true;
}

void Application::Run() {
    mTimer.On();
    while (!mWindow.ShouldClose()) {
        if (mTimer.IsFPSUpdated()) {
            mWindow.UpdateWindowTitleWithFPS(mTimer.GetFPS());
            mTimer.SetFPSUpdated(false);
        }

        if (mInputHandler.isComputeOn()) {
            mParticleSystem.Update(mProgramConfig.mGravityCenter);
        }

        mRenderer.Clear();
        mRenderer.Draw(mParticleSystem.GetCount());

        mWindow.SwapBuffer();
        mWindow.PollEvents();

        mTimer.Update();
    }
}
