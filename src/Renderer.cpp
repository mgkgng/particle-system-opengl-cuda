#include "Renderer.hpp"

Renderer::Renderer(Camera* camera) : mCamera(camera) {
    std::cout << "Renderer Initialization start." << std::endl;
    mShader = std::make_unique<Shader>("particles");
    mVAO = std::make_unique<VertexArrayObject>();
}

void Renderer::SetFramebufferSize(const int width, const int height) {
    mFramebufferWidth = width;
    mFramebufferHeight = height;
}

void Renderer::Draw(size_t particleNb) {
    mShader->Use();
    mVAO->Bind();

    if (mCamera->IsUpdated()) {
        mShader->SetUniform("uProjView", mCamera->GetViewProjMatrix());
        mCamera->SetUpdated(false);
    }

    glDrawArrays(GL_POINTS, 0, particleNb);
}