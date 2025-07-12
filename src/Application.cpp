#include "Application.hpp"

Application::Application(const int width, const int height, const char* title)
 : mWindow(width, height, title) {

    int version = gladLoadGL();
    if (!version) throw std::runtime_error("Failed to initialize OpenGL context with GLAD");

    int framebufferWidth, framebufferHeight;
    glfwGetFramebufferSize(mWindow.GetWindow(), &framebufferWidth, &framebufferHeight);

    mRenderer.InitBuffers();
    mRenderer.SetViewport(0, 0, framebufferWidth, framebufferHeight);
    mRenderer.SetFramebufferSize(framebufferWidth, framebufferHeight);

    mWindow.SetWindowUserPointer(&mInputHandler);

    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL version: "   << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
}

void Application::Run() {
    while (!mWindow.ShouldClose()) {
        glClearColor(0.3f, 0.75f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        mRenderer.Draw();

        mWindow.SwapBuffer();
        mWindow.PollEvents();
    }
}