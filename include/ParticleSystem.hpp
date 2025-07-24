#pragma once

#include <random>

#include "BufferObject.hpp"
#include "Shader.hpp"
#include "CudaComputeManager.hpp"
#include "ProgramConfig.hpp"
#include "Timer.hpp"
#include "Color.hpp"

#include "particle.cuh"

# define M_PI 3.14159265358979323846

class Window;

class ParticleSystem {
public:
    static constexpr std::string_view kDefaultColorParticle = "#6A82FB";
    static constexpr std::string_view kDefaultColorLight = "#E6007E";

    ParticleSystem(size_t particlesNb, ShapeMode shapeMode, Window* window, Timer* timer);

    void Update(const GravityCenter& gravityCenter);
    void UpdateInitialPosition();
    void Restart(ShapeMode shapeMode);

    size_t GetCount() const { return mParticlesNb; }

    void SwitchComputeOn() { mComputeOn = !mComputeOn; }
    bool IsComputeOn() const { return mComputeOn; }
    
    void SetLightPosition(const float3& pos) { mLightPos = pos; }
    const glm::vec3& GetLightPosition() const { return glm::vec3(mLightPos.x, mLightPos.y, mLightPos.z); }

private:
    static void InitializeCube(Particle** particles, size_t count, Color& particleColor, Color& lightColor);
    static void InitializeSphere(Particle** particles, size_t count, Color& particleColor, Color& lightColor);

    size_t mParticlesNb;
    std::unique_ptr<BufferObject> mSSBO;
    std::unique_ptr<CudaComputeManager> mCudaComputeManager;
    Window* mWindow;
    Timer* mTimer;
    Color mParticleColor;
    Color mLightColor;
    float3 mLightPos = { 0.0f, 0.0f, 1.0f };
    bool mComputeOn = true;
};