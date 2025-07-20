#include "particle.cuh"

__device__ float3 ApplyGravity(float3 position, GravityCenter gravityCenter) {
    float3 direction;
    direction.x = gravityCenter.position.x - position.x;
    direction.y = gravityCenter.position.y - position.y;
    direction.z = gravityCenter.position.z - position.z;

    float distanceSq = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z + 1e-6f; // avoid division by zero
    float invDistance = rsqrtf(distanceSq);
    direction.x *= invDistance;
    direction.y *= invDistance;
    direction.z *= invDistance;

    return { direction.x * gravityCenter.strength, 
             direction.y * gravityCenter.strength, 
             direction.z * gravityCenter.strength };
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

    if (gravityCenter.mode == GravityMode::Static) {
        p.velocity += ApplyGravity(p.position, gravityCenter);

        const float dt = 0.001f;
        p.position.x += p.velocity.x * dt;
        p.position.y += p.velocity.y * dt;
        p.position.z += p.velocity.z * dt;

    } else if (gravityCenter.mode == GravityMode::Off) {
        p.position = ApplyPerlin(p.initialPosition, elapsedTime);
    }

}

__host__ void LaunchUpdateParticles(Particle* particles, GravityCenter gravityCenter, int count, float elapsedTime) {
    int threadsPerBlock = 256;
    int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
    UpdateParticles<<<blocks, threadsPerBlock>>>(particles, gravityCenter, count, elapsedTime);
    // cudaDeviceSynchronize();
}
