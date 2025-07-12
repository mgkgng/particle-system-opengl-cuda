#pragma once

#include "BufferObject.hpp"
#include "Shader.hpp"

#include <random>

constexpr size_t MAX_PARTICLE_NBS = 500000;

struct Particle {
    glm::vec4 position;
    glm::vec4 velocity;
    glm::vec4 color;
    float lifespan;
    float _pad1, pad2, pad3;
};

class ParticleSystem {
public:
    ParticleSystem(size_t maxParticleNb);

    void Emit();
    void Update();

    size_t GetCount() const { return mMaxParticleNb; }

private:
    size_t mMaxParticleNb;
    std::unique_ptr<ComputeShader> mComputeShader;
    std::unique_ptr<BufferObject> mSSBO;
};