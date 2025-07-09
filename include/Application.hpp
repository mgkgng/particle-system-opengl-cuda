#pragma once

#include "Renderer.hpp"

class Application {

public:
Application(const int width, const int height, const char *title);
~Application() { glfwTerminate(); }

Application(const Application& other) = delete;
Application& operator=(const Application& other) = delete;

void Run();

private:
    Window mWindow;
    Renderer mRenderer;
};