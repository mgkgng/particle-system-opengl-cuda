#pragma once

struct Particle {
    float4 position;
    float4 velocity;
    float4 color;
    float lifespan;
    float _pad1, _pad2, _pad3;
};

__global__ void UpdateParticles(Particle* particles, int count);
void LaunchUpdateParticles(Particle* d_particles, int count);