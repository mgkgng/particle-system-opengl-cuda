#include "Window.hpp"

Window::Window(const int width, const int height, const char* title) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    mWindow = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!mWindow) {
        glfwTerminate();
        throw std::runtime_error("Failed to create windowd with GLFW.");
    }

    glfwMakeContextCurrent(mWindow);

    glfwSetWindowCloseCallback(mWindow, InputHandler::windowCloseCallback);
    glfwSetKeyCallback(mWindow, InputHandler::keyCallback);
    glfwSetMouseButtonCallback(mWindow, InputHandler::mouseButtonCallback);
    glfwSetCursorPosCallback(mWindow, InputHandler::cursorPosCallback);
}

void Window::PollEvents() {
    glfwPollEvents();

    if (glfwGetKey(mWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(mWindow, GLFW_TRUE);
    }
}
