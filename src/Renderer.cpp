#include "Renderer.hpp"

Renderer::Renderer(GLFWwindow* window, Camera* camera) : mCamera(camera) {
    std::cout << "Renderer Initialization start." << std::endl;
    mShader = std::make_unique<Shader>("particles");
    mVAO = std::make_unique<VertexArrayObject>();

    glfwGetFramebufferSize(window, &mFramebufferWidth, &mFramebufferHeight);
    glViewport(0, 0, mFramebufferWidth, mFramebufferHeight);
}

void Renderer::Clear() {
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
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