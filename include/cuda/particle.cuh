#pragma once

#include "perlin.cuh"

#include "ParticleTypes.hpp"

__host__ void cudaUpdateParticles(Particle* particles, GravityCenter gravityCenter, int count, float elapsedTime);
__host__ void cudaUpdateInitialPosition(Particle* particles, int count);