#pragma once

#include "gl_common.hpp"
#include "Window.hpp"

class Renderer {

public:
    Renderer() = default;

    void SetViewport(const int x, const int y, const int width, const int height) { glViewport(x, y, width, height); }
    void SetFramebufferSize(const int width, const int height);

private:
    int mFramebufferWidth, mFramebufferHeight;
};