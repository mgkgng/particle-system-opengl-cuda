#include "particle.cuh"

__device__ float3 ApplyGravity(float3 position, float3 velocity, GravityCenter gravityCenter) {
    const float MIN_DIST = 11.5f;
    const float PARTICLE_MASS = 100.0f;
    const float GRAVITY = 250.0f * 10.0f;

    float3 direction = gravityCenter.position - position;
    float dist = length(direction);

    if (dist < MIN_DIST) dist = MIN_DIST;

    float3 force = GRAVITY * normalize(direction) / (dist * dist + 1e-3f);
    float3 acceleration = force / PARTICLE_MASS;
    
    return velocity + acceleration;
}

__device__ float3 ApplyPerlin(float3 initialPosition, float elapsedTime) {
    float3 offset;

    offset.x = fbm(initialPosition + make_float3(elapsedTime, 0, 0));
    offset.y = fbm(initialPosition + make_float3(0, elapsedTime, 0));
    offset.z = fbm(initialPosition + make_float3(0, 0, elapsedTime));
    
    float scale = 0.3f;
    return initialPosition + offset * scale;
}

__global__ void UpdateParticles(Particle* particles, GravityCenter gravityCenter, int count, float elapsedTime) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;

    Particle& p = particles[id];

    if (gravityCenter.mode == GravityMode::Off) {
        p.position = ApplyPerlin(p.initialPosition, elapsedTime);
    } else {
        p.velocity = ApplyGravity(p.position, p.velocity, gravityCenter);

        const float dt = 0.0001f;
        p.position.x += p.velocity.x * dt;
        p.position.y += p.velocity.y * dt;
        p.position.z += p.velocity.z * dt;
    }
}

__global__ void UpdateInitialPosition(Particle* particles, int count) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;

    Particle& p = particles[id];
    p.initialPosition = p.position;
}

__host__ void cudaUpdateParticles(Particle* particles, GravityCenter gravityCenter, int count, float elapsedTime) {
    int threadsPerBlock = 256;
    int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
    UpdateParticles<<<blocks, threadsPerBlock>>>(particles, gravityCenter, count, elapsedTime);
    // cudaDeviceSynchronize();
}

__host__ void cudaUpdateInitialPosition(Particle* particles, int count) {
    int threadsPerBlock = 256;
    int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
    UpdateInitialPosition<<<blocks, threadsPerBlock>>>(particles, count);
}
