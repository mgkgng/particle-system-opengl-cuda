#include "InputHandler.hpp"
#include "Camera.hpp"
#include "Application.hpp"

void InputHandler::onKey(int key, int scancode, int action, int mods) {
    if (!mCamera || action != GLFW_PRESS) return;

    switch (key) {
        case GLFW_KEY_B:
            if (mProgramConfig->mGravityCenter.mode == GravityMode::Static || mProgramConfig->mGravityCenter.mode == GravityMode::Follow) {
                mProgramConfig->mGravityCenter.mode = GravityMode::Off;
                mParticleSystem->UpdateInitialPosition();
                mTimer->Reset();
            } else if (mProgramConfig->mGravityFollow) {
                mProgramConfig->mGravityCenter.mode = GravityMode::Follow;
            } else {
                mProgramConfig->mGravityCenter.mode = GravityMode::Static;
            }
            break;
        case GLFW_KEY_C:
            break;
        case GLFW_KEY_R:
            mParticleSystem->Restart(mProgramConfig->mShapeMode);
            mTimer->Reset();
            break;
        case GLFW_KEY_P:
            if (mProgramConfig->mGravityCenter.mode != GravityMode::Static) return;
            
            auto cursorPos = mWindow->GetCurrentCursorPos();
            mProgramConfig->mGravityCenter.position = ScreenToWorld(cursorPos[0], cursorPos[1], Application::kWindowWidth, Application::kWindowHeight, mCamera->GetViewMatrix(), mCamera->GetProjMatrix(), mCamera->GetPosition());
            break;
        case GLFW_KEY_H:
            break;
        case GLFW_KEY_SPACE:
            mParticleSystem->SwitchComputeOn();
            break;
        default:
            return;
    }
}

void InputHandler::onMouseButton(int button, int action, int mods) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;

    if (action == GLFW_PRESS) {
        mIsMouseDown = true;
        mPrevCursorPos = mWindow->GetCurrentCursorPos();
    } else if (action == GLFW_RELEASE) {
        mIsMouseDown = false;
    }
}

void InputHandler::onCursorPos(double xpos, double ypos) {
    if (mIsMouseDown) {
        mCamera->Rotate(static_cast<float>(xpos - mPrevCursorPos[0]), static_cast<float>(ypos - mPrevCursorPos[1]));
    } else if (mProgramConfig->mGravityCenter.mode == GravityMode::Follow) {
        auto cursorPos = mWindow->GetCurrentCursorPos();
        mProgramConfig->mGravityCenter.position = ScreenToWorld(cursorPos[0], cursorPos[1], Application::kWindowWidth, Application::kWindowHeight, mCamera->GetViewMatrix(), mCamera->GetProjMatrix(), mCamera->GetPosition());
    }

    SetLightPosition();
}

void InputHandler::onScroll(double xoffset, double yoffset) {
    if (yoffset == 0.0f) return;

    mCamera->Zoom(yoffset / 50.0f);
}

void InputHandler::onCursorEnter(int entered) {
    mWindow->SetCursorOnWindow(static_cast<bool>(entered));
    SetLightPosition();
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

void InputHandler::cursorEnterCallback(GLFWwindow* window, int entered) {
    auto* input = static_cast<InputHandler*>(glfwGetWindowUserPointer(window));
    if (input) input->onCursorEnter(entered);
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

void InputHandler::SetLightPosition() {
    auto cursorPos = mWindow->GetCurrentCursorPos();
    auto lightPos = ScreenToWorld(cursorPos[0], cursorPos[1], Application::kWindowWidth, Application::kWindowHeight, mCamera->GetViewMatrix(), mCamera->GetProjMatrix(), mCamera->GetPosition());
    mParticleSystem->SetLightPosition(lightPos);
}