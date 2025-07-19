#pragma once

struct Particle {
    float4 position;
    float4 velocity;
    float4 color;
    float lifespan;
    float size;
    float _pad1, _pad2;
};

struct GravityCenter {
    float3 position;
    float strength;
};

__host__ void LaunchUpdateParticles(Particle* particles, GravityCenter gravityCenter, int count);