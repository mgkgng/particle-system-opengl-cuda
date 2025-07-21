#pragma once

#include <array>
#include <iostream>

#include "gl_common.hpp"
#include "ProgramConfig.hpp"
#include "ParticleSystem.hpp"

class Camera;

class InputHandler {
public:
    InputHandler(Window* window, Camera* camera, ProgramConfig* programConfig, ParticleSystem* particleSystem, Timer* timer) 
        : mWindow(window), mCamera(camera), mProgramConfig(programConfig), mParticleSystem(particleSystem), mTimer(timer) {}

    void onKey(int key, int scancode, int action, int mods);
    void onMouseButton(GLFWwindow* window, int button, int action, int mods);
    void onCursorPos(double xpos, double ypos);
    void onScroll(double xoffset, double yoffset);

    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void windowCloseCallback(GLFWwindow* window);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    void SetCamera(Camera* camera) { mCamera = camera; }

private:
    static float3 ScreenToWorld(const double mouseX, const double mouseY,
                                const int screenWidth, const int screenHeight,
                                const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix,
                                const glm::vec3& cameraPos);

    Window* mWindow = nullptr;
    Camera* mCamera = nullptr;
    ProgramConfig* mProgramConfig = nullptr;
    ParticleSystem* mParticleSystem = nullptr;
    Timer* mTimer = nullptr;

    bool mIsMouseDown = false;
    std::array<double, 2> mPrevCursorPos;

};