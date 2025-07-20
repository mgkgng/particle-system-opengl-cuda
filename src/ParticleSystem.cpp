#include "ParticleSystem.hpp"
#include "Application.hpp"

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> cube_dis(-0.5, 0.5);
std::uniform_real_distribution<float> sphere_dis(0.0, 1.0);
std::uniform_real_distribution<float> color_dis(0.0, 1.0);


ParticleSystem::ParticleSystem(size_t particlesNb, ShapeMode shapeMode, GravityMode gravityMode, Timer *timer) : mParticlesNb(particlesNb), mTimer(timer) {
    mSSBO = std::make_unique<BufferObject>(GL_SHADER_STORAGE_BUFFER);
    mSSBO->InitializeData(nullptr, sizeof(Particle) * mParticlesNb);
    mSSBO->BindIndexedTarget(0);

    Particle* particles = static_cast<Particle*>(mSSBO->MapBuffer(GL_WRITE_ONLY));
    if (shapeMode == ShapeMode::Cube) InitializeCube(&particles, mParticlesNb);
    else InitializeSphere(&particles, mParticlesNb);
    mSSBO->UnmapBuffer();

    mCudaComputeManager = std::make_unique<CudaComputeManager>();
    mCudaComputeManager->RegisterBuffer(mSSBO->GetID());

    mGravityCenter.position = make_float3(0.0f, 0.0f, 0.0f);
    mGravityCenter.strength = 0.6f;
    mGravityCenter.mode = gravityMode;
}

void ParticleSystem::Emit() {}

void ParticleSystem::Update() {
    void *cudaResourcePtr = mCudaComputeManager->MapBuffer();
    Particle* particles = static_cast<Particle*>(cudaResourcePtr);

    LaunchUpdateParticles(particles, mGravityCenter, mParticlesNb, mTimer->GetElapsedTime());

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "[CUDA] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    // }

    // cudaDeviceSynchronize();

    mCudaComputeManager->Unmap();
}

void ParticleSystem::InitializeCube(Particle** particles, size_t count) {
    for (size_t i = 0; i < count; i++) {
        const float x = cube_dis(gen);
        const float y = cube_dis(gen);
        const float z = cube_dis(gen);

        const float r = color_dis(gen);
        const float g = color_dis(gen);
        const float b = color_dis(gen);

        (*particles)[i].position = make_float3(x, y, z);
        (*particles)[i].initialPosition = make_float3(x, y, z);
        (*particles)[i].velocity = make_float3(x * 2, y * 2, z * 2);
        (*particles)[i].color = make_float4(r, g, b, 1.0f);
    }
}

void ParticleSystem::InitializeSphere(Particle** particles, size_t count) {
    constexpr float maximumRadius = 0.3f;

    for (size_t i = 0; i < count; i++) {
        const float theta = sphere_dis(gen) * M_PI * 2.0;
        const float phi = acos(2.0 * sphere_dis(gen) - 1.0); 
        const float radialDistance = maximumRadius * cbrtf(sphere_dis(gen));

        const float x = radialDistance * sin(phi) * cos(theta);
        const float y = radialDistance * sin(phi) * sin(theta);
        const float z = radialDistance * cos(phi);

        const float r = color_dis(gen);
        const float g = color_dis(gen);
        const float b = color_dis(gen);

        (*particles)[i].position = make_float3(x, y, z);
        (*particles)[i].initialPosition = make_float3(x, y, z);
        (*particles)[i].velocity = make_float3(x * 2, y * 2, z * 2);
        (*particles)[i].color = make_float4(r, g, b, 1.0f);
    }
}
