#pragma once

#include <string>
#include <iostream>

#include "particle.cuh"

class ProgramConfig {
public:
    static constexpr float kGravityForceDefault = 0.01f;

    bool ParseArg(int argc, char **argv);

private:
    friend class Application;
    friend class InputHandler;
    friend class ImGuiLayer;

    ShapeMode mShapeMode = ShapeMode::Sphere;
    GravityCenter mGravityCenter = { make_float3(0.0f, 0.0f, 0.0f), kGravityForceDefault, GravityMode::Off };
    bool mGravityFollow = false;
    size_t mParticleCount = 1000000;
    bool mShowImGui = false;
};