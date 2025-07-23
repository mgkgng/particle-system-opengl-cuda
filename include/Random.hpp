#pragma once

#include <random>

namespace Random {
    extern std::mt19937 gen;

    extern std::uniform_real_distribution<float> color_dis;
    extern std::uniform_real_distribution<float> cube_dis;
    extern std::uniform_real_distribution<float> sphere_dis;

    float RandomColor();
    float RandomCube();
    float RandomSphere();
}
