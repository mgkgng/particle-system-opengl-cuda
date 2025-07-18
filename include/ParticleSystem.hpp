#pragma once

#include "BufferObject.hpp"
#include "Shader.hpp"
#include "CudaComputeManager.hpp"
#include "particle.cuh"
#include <cuda_runtime.h>
#include <random>

constexpr size_t MAX_PARTICLE_NBS = 500000;

class ParticleSystem {
public:
    ParticleSystem(size_t maxParticleNb);

    void Emit();
    void Update();

    size_t GetCount() const { return mMaxParticleNb; }

private:
    size_t mMaxParticleNb;
    std::unique_ptr<BufferObject> mSSBO;
    std::unique_ptr<CudaComputeManager> mCudaComputeManager;
    GravityCenter mGravityCenter;
};