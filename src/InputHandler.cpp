#include "InputHandler.hpp"

void InputHandler::onKey(int key, int scancode, int action, int mods) {
    std::cout << "onKey " << key << " " << scancode << " " << action << " " << mods << std::endl;
}

void InputHandler::onMouseButton(int button, int action, int mods) {
    std::cout << "mouse button " << button << " " << action << " " << mods << std::endl;
}

void InputHandler::onCursorPos(double xpos, double ypos) {
    std::cout << "curos pos " << xpos << " " << ypos << std::endl;
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



