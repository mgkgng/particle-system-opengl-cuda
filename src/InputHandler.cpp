#include "InputHandler.hpp"
#include "Camera.hpp"

void InputHandler::onKey(int key, int scancode, int action, int mods) {
    if (!mCamera) return;

    switch (key) {
        case GLFW_KEY_W:
            std::cout << "W ";
            break;
        case GLFW_KEY_A:
            std::cout << "A ";
            break;
        case GLFW_KEY_S:
            std::cout << "S ";
            break;
        case GLFW_KEY_D:
            std::cout << "D ";
            break;
        default:
            return;
    }

    switch (action) {
        case GLFW_PRESS:
            std::cout << "PRESS" << std::endl;
            break;
        case GLFW_REPEAT:
            std::cout << "REPEAT" << std::endl;
            break;
        case GLFW_RELEASE:
            std::cout << "RELEASE" << std::endl;
            break;
        default:
            return;
    }
}

void InputHandler::onMouseButton(int button, int action, int mods) {
    // std::cout << "mouse button " << button << " " << action << " " << mods << std::endl;
}

void InputHandler::onCursorPos(double xpos, double ypos) {
    // std::cout << "curos pos " << xpos << " " << ypos << std::endl;
}

void InputHandler::onScroll(double xoffset, double yoffset) {
    if (yoffset == 0.0f) return;

    mCamera->Zoom(yoffset / 50.0f);
}

void InputHandler::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    auto* input = static_cast<InputHandler*>(glfwGetWindowUserPointer(window));
    if (input) input->onKey(key, scancode, action, mods);
}

void InputHandler::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    auto* input = static_cast<InputHandler*>(glfwGetWindowUserPointer(window));
    if (input) input->onMouseButton(button, action, mods);
}

void InputHandler::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    auto* input = static_cast<InputHandler*>(glfwGetWindowUserPointer(window));
    if (input) input->onCursorPos(xpos, ypos);
}

void InputHandler::windowCloseCallback(GLFWwindow* window) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
}

void InputHandler::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    auto* input = static_cast<InputHandler*>(glfwGetWindowUserPointer(window));
    if (input) input->onScroll(xoffset, yoffset); 
}




