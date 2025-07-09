#pragma once

#include "gl_common.hpp"
#include "Window.hpp"
#include "BufferObject.hpp"
#include "VertexArrayObject.hpp"
#include "Shader.hpp"

constexpr float triangleVertices[] = {
    //   x      y      z       r     g     b
    -0.5f, -0.5f, 0.0f,   1.0f, 0.0f, 0.0f, // bottom left
     0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f, // bottom right
     0.0f,  0.5f, 0.0f,   0.0f, 0.0f, 1.0f  // top center
};

constexpr unsigned int triangleIndices[] = {
    0, 1, 2
};

class Renderer {

public:
    Renderer();

    void Draw();
    void InitBuffers();

    void SetViewport(const int x, const int y, const int width, const int height) { glViewport(x, y, width, height); }
    void SetFramebufferSize(const int width, const int height);

private:
    int mFramebufferWidth, mFramebufferHeight;

    std::unique_ptr<VertexArrayObject> mVAO;
    std::unique_ptr<BufferObject> mVBO;
    std::unique_ptr<BufferObject> mEBO;
    std::unique_ptr<Shader> mShader;
};