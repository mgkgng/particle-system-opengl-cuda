#pragma once

#include "float3_utils.cuh"

#define NUM_NOISE_OCTAVES 5

__device__ float fbm(float3 x);