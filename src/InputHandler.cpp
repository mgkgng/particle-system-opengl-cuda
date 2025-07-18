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
        case GLFW_KEY_SPACE:
            if (action == GLFW_PRESS)
                mComputeOn = !mComputeOn;
            break;
        default:
            return;
    }

    //     case GLFW_PRESS:
    //     case GLFW_REPEAT:
    //     case GLFW_RELEASE:
}

void InputHandler::onMouseButton(GLFWwindow* window, int button, int action, int mods) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;

    if (action == GLFW_PRESS) {
        mIsMouseDown = true;
        glfwGetCursorPos(window, &mPrevCursorPos[0], &mPrevCursorPos[1]);
    } else if (action == GLFW_RELEASE) {
        mIsMouseDown = false;
    }
}

void InputHandler::onCursorPos(double xpos, double ypos) {
    if (!mIsMouseDown) return;

    mCamera->Rotate(static_cast<float>(xpos - mPrevCursorPos[0]), static_cast<float>(ypos - mPrevCursorPos[1]));
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
    if (input) input->onMouseButton(window, button, action, mods);
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
