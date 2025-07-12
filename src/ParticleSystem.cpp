#include "ParticleSystem.hpp"

ParticleSystem::ParticleSystem(size_t maxParticleNb) : mMaxParticleNb(maxParticleNb) {
    mSSBO = std::make_unique<BufferObject>(GL_SHADER_STORAGE_BUFFER);
    mSSBO->InitializeData(nullptr, sizeof(Particle) * mMaxParticleNb);
    mSSBO->BindIndexedTarget(0);

    GLint maxWorkGroupSize[3];
    for (size_t i = 0; i < 3; ++i) {
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, i, &maxWorkGroupSize[i]);
        std::cout << "max work group size for index " << i << ": " << maxWorkGroupSize[i] << std::endl;
    }
}

void ParticleSystem::Emit() {}

void ParticleSystem::Update() {}