#pragma once

#include "BufferObject.hpp"
#include "Shader.hpp"

constexpr size_t MAX_PARTICLE_NBS = 500000;

struct Particle {
    glm::vec3 pos;
    glm::vec3 velocity;
    float lifetime;
    
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