#include "Renderer.hpp"

Renderer::Renderer() {
    std::cout << "Renderer Initialization start." << std::endl;
    mShader = std::make_unique<Shader>("particles");

}

// void Renderer::InitBuffers() {
//     mShader = std::make_unique<Shader>("test");
//     mVAO = std::make_unique<VertexArrayObject>();
//     mVBO = std::make_unique<BufferObject>(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
//     mEBO = std::make_unique<BufferObject>(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);

//     mVAO->Bind();
//     mVBO->InitializeData(&triangleVertices[0], sizeof(triangleVertices));
//     mEBO->InitializeData(&triangleIndices[0], sizeof(triangleIndices));

//     mVAO->SetAttribute(0, 3, GL_FLOAT, GL_FALSE, 24, 0);
//     mVAO->SetAttribute(1, 3, GL_FLOAT, GL_FALSE, 24, (void*) 12);
    
//     mVAO->Unbind();
//     mVBO->Unbind();
// }

void Renderer::SetFramebufferSize(const int width, const int height) {
    mFramebufferWidth = width;
    mFramebufferHeight = height;
}

void Renderer::Draw(size_t particleNb) {
    mShader->Use();
    // mVAO->Bind();
    // mEBO->Bind();
    glDrawArrays(GL_POINTS, 0, particleNb);
}