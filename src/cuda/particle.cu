#include "particle.cuh"

__global__ void UpdateParticles(Particle* particles, GravityCenter gravityCenter, int count) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;

    Particle& p = particles[id];

    float3 direction;
    direction.x = gravityCenter.position.x - p.position.x;
    direction.y = gravityCenter.position.y - p.position.y;
    direction.z = gravityCenter.position.z - p.position.z;

    float distanceSq = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z + 1e-6f; // avoid division by zero
    float invDistance = rsqrtf(distanceSq);
    direction.x *= invDistance;
    direction.y *= invDistance;
    direction.z *= invDistance;

    p.velocity.x += direction.x * gravityCenter.strength;
    p.velocity.y += direction.y * gravityCenter.strength;
    p.velocity.z += direction.z * gravityCenter.strength;

    float dt = 0.01f;
    p.position.x += p.velocity.x * dt;
    p.position.y += p.velocity.y * dt;
    p.position.z += p.velocity.z * dt;
}

__host__ void LaunchUpdateParticles(Particle* particles, GravityCenter gravityCenter, int count) {
    int threadsPerBlock = 256;
    int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
    UpdateParticles<<<blocks, threadsPerBlock>>>(particles, gravityCenter, count);

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "[CUDA] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    // }
    // cudaDeviceSynchronize();
}
