#pragma once

#include "gl_common.hpp"
#include <cuda_gl_interop.h>

class CudaComputeManager {
public:
    CudaComputeManager() = default;
    ~CudaComputeManager() { cudaGraphicsUnregisterResource(mCudaSSBO); }

    void RegisterBuffer(GLuint bufferId) {
        cudaGraphicsGLRegisterBuffer(&mCudaSSBO, bufferId, cudaGraphicsMapFlagsNone);
    }

    void* MapBuffer() {
        void* ptr = nullptr;
        size_t size;
        
        cudaGraphicsMapResources(1, &mCudaSSBO, 0);
        cudaGraphicsResourceGetMappedPointer(&ptr, &size, mCudaSSBO);

        return ptr;
    }

    void Unmap() {
        cudaGraphicsUnmapResources(1, &mCudaSSBO, 0);
    }

    void LaunchParticleKernel();

private:
    cudaGraphicsResource* mCudaSSBO = nullptr;
};