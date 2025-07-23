#pragma once

#include <cuda_runtime.h>
#include <string>

#include "Random.hpp"

class Color {
public:
    Color(const std::string& hex);

    float4 Perturb(float variance = 0.35f);
private:
    float3 mColor;
};
