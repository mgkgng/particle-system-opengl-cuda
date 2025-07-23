#include "Renderer.hpp"

Renderer::Renderer(Window* window, Camera* camera) : mWindow(window), mCamera(camera) {
    std::cout << "Renderer Initialization start." << std::endl;
    mParticleShader = std::make_unique<Shader>("particles");
    mCursorShader = std::make_unique<Shader>("cursor");
    mVAO = std::make_unique<VertexArrayObject>();

    glfwGetFramebufferSize(window->GetWindow(), &mFramebufferWidth, &mFramebufferHeight);
    glViewport(0, 0, mFramebufferWidth, mFramebufferHeight);
}

void Renderer::Clear() {
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

void Renderer::Draw(size_t particleNb) {
    mParticleShader->Use();
    mVAO->Bind();

    if (mCamera->IsUpdated()) {
        mParticleShader->SetUniform("uProjView", mCamera->GetViewProjMatrix());
        mCamera->SetUpdated(false);
    }

    glDrawArrays(GL_POINTS, 0, particleNb);

    if (mWindow->IsCursorOnWindow()) {
        mCursorShader->Use();
        auto cursorPos = mWindow->GetCurrentCursorPosNDC();
        mCursorShader->SetUniform("uCursorPos", cursorPos[0], cursorPos[1]);
        glDrawArrays(GL_POINTS, 0, 1);
    }
}