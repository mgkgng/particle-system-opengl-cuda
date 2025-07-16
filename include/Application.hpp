#pragma once

#include "Renderer.hpp"
#include "InputHandler.hpp"
#include "ParticleSystem.hpp"
#include "Camera.hpp"

#include <cuda_runtime.h>

class Application {

public:
Application(const int width, const int height, const char *title);
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
};