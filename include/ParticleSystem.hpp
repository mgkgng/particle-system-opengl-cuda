#pragma once

#include <random>
#include <cuda_runtime.h>

#include "BufferObject.hpp"
#include "Shader.hpp"
#include "CudaComputeManager.hpp"
#include "ProgramConfig.hpp"
#include "Timer.hpp"

#include "particle.cuh"

# define M_PI 3.14159265358979323846

class ParticleSystem {
public:
    ParticleSystem(size_t particlesNb, ShapeMode shapeMode, GravityMode gravityMode, Timer* timer);

    void Emit();
    void Update();

    size_t GetCount() const { return mParticlesNb; }

private:
    static void InitializeCube(Particle** particles, size_t count);
    static void InitializeSphere(Particle** particles, size_t count);

    size_t mParticlesNb;
    std::unique_ptr<BufferObject> mSSBO;
    std::unique_ptr<CudaComputeManager> mCudaComputeManager;
    GravityCenter mGravityCenter;
    Timer* mTimer;
};