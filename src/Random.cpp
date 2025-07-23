#include "Random.hpp"

namespace Random {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> color_dis(0.0f, 1.0f);
    std::uniform_real_distribution<float> cube_dis(-0.5f, 0.5f);
    std::uniform_real_distribution<float> sphere_dis(0.0f, 1.0f);

    float RandomColor() {
        return color_dis(gen);
    }

    float RandomCube() {
        return cube_dis(gen);
    }

    float RandomSphere() {
        return sphere_dis(gen);
    }
}
