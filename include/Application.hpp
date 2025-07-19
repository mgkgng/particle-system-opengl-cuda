#pragma once

#include <cuda_runtime.h>

#include "Renderer.hpp"
#include "InputHandler.hpp"
#include "ParticleSystem.hpp"
#include "ProgramConfig.hpp"
#include "Camera.hpp"
#include "Timer.hpp"

class Application {

public:
    static constexpr int kWindowWidth = 1280;
    static constexpr int kWindowHeight = 720;
    static constexpr float kAspectRatio = static_cast<float>(kWindowWidth) / static_cast<float>(kWindowHeight);
    static constexpr std::string_view kWindowTitle = "Particle System";

    Application(ProgramConfig programConfig);
    ~Application() { glfwTerminate(); }

    Application(const Application& other) = delete;
    Application& operator=(const Application& other) = delete;

    bool InitCUDA();
    void Run();

private:
    Window mWindow;
    Camera mCamera;
    Renderer mRenderer;
    InputHandler mInputHandler;
    ParticleSystem mParticleSystem;
    Timer mTimer;
    
};
