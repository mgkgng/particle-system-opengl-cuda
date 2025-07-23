#include "Color.hpp"

Color::Color(const std::string& hex) {
    unsigned int rgb = std::stoul(hex.substr(1), nullptr, 16);
    float r = ((rgb >> 16) & 0xFF) / 255.0f;
    float g = ((rgb >> 8) & 0xFF) / 255.0f;
    float b = (rgb & 0xFF) / 255.0f;
    mColor = make_float3(r, g, b);
}

float4 Color::Perturb(float variance) {
    auto clamp = [](float v) { return std::min(1.0f, std::max(0.0f, v)); };
    return make_float4(
        clamp(mColor.x + variance * (2.0f * Random::RandomColor() - 1.0f)),
        clamp(mColor.y + variance * (2.0f * Random::RandomColor() - 1.0f)),
        clamp(mColor.z + variance * (2.0f * Random::RandomColor() - 1.0f)),
        1.0f
    );
}
