#include "particle.cuh"

__global__ void UpdateParticles(Particle* particles, int count) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;

    particles[id].position.x += particles[id].velocity.x * 0.01;
    particles[id].position.y += particles[id].velocity.y * 0.01;
    particles[id].position.z += particles[id].velocity.z * 0.01;
}

void LaunchUpdateParticles(Particle* d_particles, int count) {
    int threadsPerBlock = 256;
    int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
    UpdateParticles<<<blocks, threadsPerBlock>>>(d_particles, count);

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "[CUDA] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    // }
    cudaDeviceSynchronize();
}
