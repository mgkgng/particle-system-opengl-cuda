#pragma once

#include "gl_common.hpp"
#include "Window.hpp"
#include "BufferObject.hpp"
#include "VertexArrayObject.hpp"
#include "Shader.hpp"
#include "Camera.hpp"

class Renderer {

public:
    Renderer(Camera* camera);

    void Draw(size_t particleNb);

    void SetViewport(const int x, const int y, const int width, const int height) { glViewport(x, y, width, height); }
    void SetFramebufferSize(const int width, const int height);
private:
    int mFramebufferWidth, mFramebufferHeight;

    std::unique_ptr<VertexArrayObject> mVAO;
    std::unique_ptr<Shader> mShader;

    Camera* mCamera;
};