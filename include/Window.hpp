#pragma once

#include "gl_common.hpp"
#include "utils.hpp"

class Window {
public:
    Window(const int width, const int height, const char* title);
    bool ShouldClose() { return false; }
    GLFWwindow* GetWindow() const { return mWindow; }
    void SwapBuffer() { glfwSwapBuffers(mWindow); }

private:
    GLFWwindow* mWindow = nullptr; 
};