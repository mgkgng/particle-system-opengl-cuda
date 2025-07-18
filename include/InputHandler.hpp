#pragma once

#include "gl_common.hpp"
#include "utils.hpp"
#include <array>

class Camera;

class InputHandler {
public:
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

    bool isComputeOn() const { return mComputeOn; }

private:
    Camera* mCamera = nullptr;
    bool mIsMouseDown = false;
    std::array<double, 2> mPrevCursorPos;

    bool mComputeOn = true;
};