#include "Window.hpp"
#include "Application.hpp"

Window::Window() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    mWindow = glfwCreateWindow(Application::kWindowWidth, Application::kWindowHeight, Application::kWindowTitle.data(), nullptr, nullptr);
    if (!mWindow) {
        glfwTerminate();
        throw std::runtime_error("Failed to create windowd with GLFW.");
    }

    glfwMakeContextCurrent(mWindow);

    glfwSetWindowCloseCallback(mWindow, InputHandler::windowCloseCallback);
    glfwSetKeyCallback(mWindow, InputHandler::keyCallback);
    glfwSetMouseButtonCallback(mWindow, InputHandler::mouseButtonCallback);
    glfwSetCursorPosCallback(mWindow, InputHandler::cursorPosCallback);
    glfwSetScrollCallback(mWindow, InputHandler::scrollCallback);

    int version = gladLoadGL();
    if (!version) throw std::runtime_error("Failed to initialize OpenGL context with GLAD");

    // Disable VSync
    glfwSwapInterval(0);

    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL version: "   << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
}

void Window::PollEvents() {
    glfwPollEvents();

    if (glfwGetKey(mWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(mWindow, GLFW_TRUE);
    }
}

void Window::UpdateWindowTitleWithFPS(float fps) {
    std::ostringstream title;
    title << "Particle System - FPS: " << std::fixed << std::setprecision(2) << fps;
    glfwSetWindowTitle(mWindow, title.str().c_str());
}
