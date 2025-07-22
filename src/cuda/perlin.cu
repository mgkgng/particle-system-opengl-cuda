#pragma once

#include "perlin.cuh"

__device__ inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ float fract(float x) {
    return x - floorf(x);
}

__device__ float hash(float p) {
    p = fract(p * 0.011f);
    p *= p + 7.5f;
    p *= p + p;
    return fract(p);
}

__device__ float hash(float2 p) {
    float3 p3 = make_float3(fract(p.x * 0.13f), fract(p.y * 0.13f), fract(p.x * 0.13f));
    float d = dot(p3, make_float3(p3.y + 3.333f, p3.z + 3.333f, p3.x + 3.333f));
    return fract((p3.x + p3.y) * p3.z);
}

__device__ float hash(float3 p) {
    float3 step = make_float3(110.0f, 241.0f, 171.0f);
    return hash(dot(p, step));
}

__device__ float noise(float3 x) {
    const float3 step = make_float3(110.0f, 241.0f, 171.0f);

    float3 i = make_float3(floorf(x.x), floorf(x.y), floorf(x.z));
    float3 f = make_float3(fract(x.x), fract(x.y), fract(x.z));

    float n = dot(i, step);
    float3 u = f * f * (make_float3(3.0f, 3.0f, 3.0f) - 2.0f * f);

    float v000 = hash(n + dot(step, make_float3(0, 0, 0)));
    float v100 = hash(n + dot(step, make_float3(1, 0, 0)));
    float v010 = hash(n + dot(step, make_float3(0, 1, 0)));
    float v110 = hash(n + dot(step, make_float3(1, 1, 0)));
    float v001 = hash(n + dot(step, make_float3(0, 0, 1)));
    float v101 = hash(n + dot(step, make_float3(1, 0, 1)));
    float v011 = hash(n + dot(step, make_float3(0, 1, 1)));
    float v111 = hash(n + dot(step, make_float3(1, 1, 1)));

    float x00 = lerp(v000, v100, u.x);
    float x10 = lerp(v010, v110, u.x);
    float x01 = lerp(v001, v101, u.x);
    float x11 = lerp(v011, v111, u.x);

    float y0 = lerp(x00, x10, u.y);
    float y1 = lerp(x01, x11, u.y);

    return lerp(y0, y1, u.z);
}

__device__ float fbm(float3 x) {
    float v = 0.0f;
    float a = 0.5f;
    float3 shift = make_float3(100.0f, 100.0f, 100.0f);

    for (int i = 0; i < NUM_NOISE_OCTAVES; ++i) {
        v += a * noise(x);
        x = x * 2.0f + shift;
        a *= 0.5f;
    }
    return v;
}
