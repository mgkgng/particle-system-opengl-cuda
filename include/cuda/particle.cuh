#pragma once

#include "perlin.cuh"

#include "ProgramConfig.hpp"

struct Particle {
    float3 position;
    float _pad1;

    float3 initialPosition;
    float _pad2;

    float3 velocity;
    float _pad3;
    
    float4 color;
    float lifespan;
    float size;
    float _padding[2];
};

struct GravityCenter {
    float3 position;
    float strength;
    GravityMode mode;
};

__host__ void LaunchUpdateParticles(Particle* particles, GravityCenter gravityCenter, int count, float elapsedTime);