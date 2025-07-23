#include "ParticleSystem.hpp"
#include "Application.hpp"

ParticleSystem::ParticleSystem(size_t particlesNb, ShapeMode shapeMode, Window *window, Timer *timer)
    : mParticlesNb(particlesNb), mWindow(window), mTimer(timer), mColor(std::string(kDefaultColor)) {
    mSSBO = std::make_unique<BufferObject>(GL_SHADER_STORAGE_BUFFER);
    mSSBO->InitializeData(nullptr, sizeof(Particle) * mParticlesNb);
    mSSBO->BindIndexedTarget(0);

    Restart(shapeMode);

    mCudaComputeManager = std::make_unique<CudaComputeManager>();
    mCudaComputeManager->RegisterBuffer(mSSBO->GetID());
}

void ParticleSystem::Restart(ShapeMode shapeMode) {
    Particle* particles = static_cast<Particle*>(mSSBO->MapBuffer(GL_WRITE_ONLY));
    if (shapeMode == ShapeMode::Cube) InitializeCube(&particles, mParticlesNb, mColor);
    else InitializeSphere(&particles, mParticlesNb, mColor);
    mSSBO->UnmapBuffer();
}

void ParticleSystem::Update(const GravityCenter& gravityCenter) {
    void *cudaResourcePtr = mCudaComputeManager->MapBuffer();
    Particle* particles = static_cast<Particle*>(cudaResourcePtr);

    cudaUpdateParticles(particles, gravityCenter, mParticlesNb, mTimer->GetElapsedTime());
    mCudaComputeManager->Unmap();
}

void ParticleSystem::UpdateInitialPosition() {
    void *cudaResourcePtr = mCudaComputeManager->MapBuffer();
    Particle* particles = static_cast<Particle*>(cudaResourcePtr);

    cudaUpdateInitialPosition(particles, mParticlesNb);
    mCudaComputeManager->Unmap();
}

void ParticleSystem::InitializeCube(Particle** particles, size_t count, Color& color) {
    for (size_t i = 0; i < count; i++) {
        const float x = Random::RandomCube();
        const float y = Random::RandomCube();
        const float z = Random::RandomCube();

        (*particles)[i].position = make_float3(x, y, z);
        (*particles)[i].initialPosition = make_float3(x, y, z);
        (*particles)[i].color = color.Perturb();
    }
}

void ParticleSystem::InitializeSphere(Particle** particles, size_t count, Color& color) {
    constexpr float maximumRadius = 0.3f;

    for (size_t i = 0; i < count; i++) {
        const float theta = Random::RandomSphere() * M_PI * 2.0;
        const float phi = acos(2.0 * Random::RandomSphere() - 1.0); 
        const float radialDistance = maximumRadius * cbrtf(Random::RandomSphere());

        const float x = radialDistance * sin(phi) * cos(theta);
        const float y = radialDistance * sin(phi) * sin(theta);
        const float z = radialDistance * cos(phi);

        (*particles)[i].position = make_float3(x, y, z);
        (*particles)[i].initialPosition = make_float3(x, y, z);
        (*particles)[i].color = color.Perturb();
    }
}
