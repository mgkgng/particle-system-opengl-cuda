#pragma once

#include "operator.cuh"

#define NUM_NOISE_OCTAVES 5

__device__ float fbm(float3 x);