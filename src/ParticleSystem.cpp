#include "ParticleSystem.hpp"

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> cube_dis(-0.5, 0.5);
std::uniform_real_distribution<float> col_dis(0.0, 1.0);

ParticleSystem::ParticleSystem(size_t maxParticleNb) : mMaxParticleNb(maxParticleNb) {
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

        particles[i].position = make_float4(x, y, z, 1.0f);
        particles[i].velocity = make_float4(x * 2, y * 2, z * 2, 0.0f);
        particles[i].color = make_float4(r, g, b, 1.0f);
        particles[i].lifespan = 100.0;
    }
    mSSBO->UnmapBuffer();

    mCudaComputeManager = std::make_unique<CudaComputeManager>();
    mCudaComputeManager->RegisterBuffer(mSSBO->GetID());
}

void ParticleSystem::Emit() {}

void ParticleSystem::Update() {
    void *cudaResourcePtr = mCudaComputeManager->MapBuffer();
    Particle* particles = static_cast<Particle*>(cudaResourcePtr);

    LaunchUpdateParticles(particles, mMaxParticleNb);

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "[CUDA] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    // }

    // cudaDeviceSynchronize();

    mCudaComputeManager->Unmap();
}