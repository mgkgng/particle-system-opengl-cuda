#pragma once

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <array>

#include "gl_common.hpp"
#include "InputHandler.hpp"

class InputHandler;

class Window {
public:
    Window();

    bool ShouldClose() { return glfwWindowShouldClose(mWindow); }
    void SwapBuffer() { glfwSwapBuffers(mWindow); }
    void PollEvents();

    GLFWwindow* GetWindow() const { return mWindow; }
    void SetWindowUserPointer(void* ptr) const { glfwSetWindowUserPointer(mWindow, ptr); }
    void UpdateWindowTitleWithFPS(float fps);
    std::array<double, 2> GetCurrentCursorPos();
    std::array<float, 2> GetCurrentCursorPosNDC();
    bool IsCursorOnWindow() const { return mIsCursorOnWindow; }
    void SetCursorOnWindow(bool entered) { mIsCursorOnWindow = entered; }

private:
    GLFWwindow* mWindow = nullptr;
    bool mIsCursorOnWindow = false;
};