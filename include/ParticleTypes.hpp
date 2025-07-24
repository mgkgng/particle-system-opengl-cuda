#pragma once

#include <cuda_runtime.h>

enum ShapeMode { Cube, Sphere };
enum GravityMode { Off = 0, Static = 1, Follow = 2 };

struct Particle {
    float3 position;
    float _pad1;

    float3 initialPosition;
    float _pad2;

    float3 velocity;
    float _pad3;
    
    float4 particleColor;
    float4 lightColor;
    
    float lifespan;
    float size;
    float _padding[2];
};

struct GravityCenter {
    float3 position;
    float strength;
    GravityMode mode;
};

