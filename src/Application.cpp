#include "Application.hpp"

Application::Application(const int width, const int height, const char* title)
 : mWindow(width, height, title)
 , mParticleSystem(MAX_PARTICLE_NBS)
 , mCamera(45.0f, static_cast<float>(width) / static_cast<float>(height), 1.0f, 100.0f)
 , mRenderer(&mCamera) {
    int framebufferWidth, framebufferHeight;
    glfwGetFramebufferSize(mWindow.GetWindow(), &framebufferWidth, &framebufferHeight);

    mRenderer.SetViewport(0, 0, framebufferWidth, framebufferHeight);
    mRenderer.SetFramebufferSize(framebufferWidth, framebufferHeight);

    mWindow.SetWindowUserPointer(&mInputHandler);

    mInputHandler.SetCamera(&mCamera);

#ifdef DEBUG_ON
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL version: "   << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
#endif
}

void Application::Run() {
    while (!mWindow.ShouldClose()) {
        mParticleSystem.Update();

        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        mRenderer.Draw(mParticleSystem.GetCount());

        mWindow.SwapBuffer();
        mWindow.PollEvents();
    }
}