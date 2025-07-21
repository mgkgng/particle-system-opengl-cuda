#include "InputHandler.hpp"
#include "Camera.hpp"
#include "Application.hpp"

void InputHandler::onKey(int key, int scancode, int action, int mods) {
    if (!mCamera) return;

    switch (key) {
        case GLFW_KEY_B:
            if (action == GLFW_PRESS) {
                if (mProgramConfig->mGravityCenter.mode == GravityMode::Static || mProgramConfig->mGravityCenter.mode == GravityMode::Follow) {
                    mProgramConfig->mGravityCenter.mode = GravityMode::Off;
                    mParticleSystem->UpdateInitialPosition();
                    mTimer->Reset();
                } else if (mProgramConfig->mGravityFollow) {
                    mProgramConfig->mGravityCenter.mode = GravityMode::Follow;
                } else {
                    mProgramConfig->mGravityCenter.mode = GravityMode::Static;
                }
            }
            break;
        case GLFW_KEY_C:
            // cursor on/off
            break;
        case GLFW_KEY_R:
            if (action != GLFW_PRESS || mProgramConfig->mGravityCenter.mode != GravityMode::Static) return;
            
            double mouseX, mouseY;
            glfwGetCursorPos(mWindow, &mouseX, &mouseY);

            mProgramConfig->mGravityCenter.position = ScreenToWorld(mouseX, mouseY, Application::kWindowWidth, Application::kWindowHeight, mCamera->GetViewMatrix(), mCamera->GetProjMatrix(), mCamera->GetPosition());
            std::cout << "let's see x: " << mProgramConfig->mGravityCenter.position.x << " y: " << mProgramConfig->mGravityCenter.position.y << " z: " << mProgramConfig->mGravityCenter.position.z << std::endl;
            break;
        case GLFW_KEY_H:
            break;
        case GLFW_KEY_SPACE:
            if (action == GLFW_PRESS)
                mComputeOn = !mComputeOn;
            break;
        default:
            return;
    }
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

float3 InputHandler::ScreenToWorld(const double mouseX, const double mouseY,
                                   const int screenWidth, const int screenHeight,
                                   const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix,
                                   const glm::vec3& cameraPos) {
    float x = (2.0f * mouseX) / screenWidth - 1.0f;
    float y = 1.0f - (2.0f * mouseY) / screenHeight;
    glm::vec4 rayClip = glm::vec4(x, y, -1.0f, 1.0f);
    
    glm::vec4 rayEye = glm::inverse(projectionMatrix) * rayClip;
    rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);

    glm::vec3 rayWorld = glm::normalize(glm::vec3(glm::inverse(viewMatrix) * rayEye));

    float t = -cameraPos.z / rayWorld.z;
    auto res = cameraPos + t * rayWorld;

    return make_float3(res.x, res.y, 0.0f);
}