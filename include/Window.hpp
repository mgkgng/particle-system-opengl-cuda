#pragma once

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>

#include "gl_common.hpp"
#include "InputHandler.hpp"

class InputHandler;

class Window {
public:
    Window(const int width, const int height, const char* title);
    bool ShouldClose() { return glfwWindowShouldClose(mWindow); }
    GLFWwindow* GetWindow() const { return mWindow; }
    void SetWindowUserPointer(void* ptr) const { glfwSetWindowUserPointer(mWindow, ptr); }
    void UpdateWindowTitleWithFPS(float fps);

    void SwapBuffer() { glfwSwapBuffers(mWindow); }
    void PollEvents();

private:
    GLFWwindow* mWindow = nullptr; 
};