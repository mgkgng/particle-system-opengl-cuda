#include "Application.hpp"

Application::Application(const int width, const int height, const char* title)
 : mWindow(width, height, title)
 , mParticleSystem(MAX_PARTICLE_NBS)
 , mCamera(45.0f, static_cast<float>(width) / static_cast<float>(height), 1.0f, 100.0f)
 , mRenderer(mWindow.GetWindow(), &mCamera) {
    mWindow.SetWindowUserPointer(&mInputHandler);
    mInputHandler.SetCamera(&mCamera);
#ifdef DEBUG_ON
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL version: "   << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
#endif
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

        mParticleSystem.Update();

        mRenderer.Clear();
        mRenderer.Draw(mParticleSystem.GetCount());

        mWindow.SwapBuffer();
        mWindow.PollEvents();

        mTimer.Update();
    }
}
