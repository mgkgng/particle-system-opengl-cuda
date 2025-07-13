#include "ParticleSystem.hpp"

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> cube_dis(-0.5, 0.5);
std::uniform_real_distribution<float> col_dis(0.0, 1.0);

ParticleSystem::ParticleSystem(size_t maxParticleNb) : mMaxParticleNb(maxParticleNb) {\
    mComputeShader = std::make_unique<ComputeShader>("particles");

    mSSBO = std::make_unique<BufferObject>(GL_SHADER_STORAGE_BUFFER);
    mSSBO->InitializeData(nullptr, sizeof(Particle) * mMaxParticleNb);
    mSSBO->BindIndexedTarget(0);

#ifdef DEBUG_ON
    GLint maxWorkGroupSize[3];
    for (size_t i = 0; i < 3; ++i) {
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, i, &maxWorkGroupSize[i]);
        std::cout << "max work group size for index " << i << ": " << maxWorkGroupSize[i] << std::endl;
    }
#endif

    Particle* particles = static_cast<Particle*>(mSSBO->MapBuffer(GL_WRITE_ONLY));
    for (size_t i = 0; i < mMaxParticleNb; i++) {
        const float x = cube_dis(gen);
        const float y = cube_dis(gen);
        const float z = cube_dis(gen);

        const float r = col_dis(gen);
        const float g = col_dis(gen);
        const float b = col_dis(gen);

        particles[i].position = glm::vec4(x, y, z, 1.0f);
        particles[i].velocity = glm::vec4(x * 2, y * 2, z * 2, 0.0f);
        particles[i].color = glm::vec4(r, g, b, 1.0f);
        particles[i].lifespan = 100.0;
    }
    mSSBO->UnmapBuffer();
}

void ParticleSystem::Emit() {}

void ParticleSystem::Update() {
    mComputeShader->Use();

    GLuint workGroupSize = 256;
    GLuint numGroups = (mMaxParticleNb + workGroupSize - 1) / workGroupSize;
    //std::cout << "num groups: " << numGroups << std::endl;
    mComputeShader->Compute(numGroups, 1, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}