#pragma once

#include "gl_common.hpp"
#include "Window.hpp"
#include "BufferObject.hpp"
#include "VertexArrayObject.hpp"
#include "Shader.hpp"
#include "Camera.hpp"

class Renderer {

public:
    Renderer(Window* window, Camera* camera);

    void Clear();
    void Draw(ParticleSystem* particleSystem);

    void SetViewport(const int x, const int y, const int width, const int height) { glViewport(x, y, width, height); }
private:
    int mFramebufferWidth, mFramebufferHeight;

    std::unique_ptr<VertexArrayObject> mVAO;
    std::unique_ptr<Shader> mParticleShader;
    std::unique_ptr<Shader> mCursorShader;

    Window *mWindow;
    Camera* mCamera;

};