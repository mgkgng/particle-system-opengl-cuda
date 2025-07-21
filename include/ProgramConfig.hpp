#pragma once

#include <string>
#include <iostream>

#include "particle.cuh"

class ProgramConfig {
public:
    bool ParseArg(int argc, char **argv);

private:
    friend class Application;
    friend class InputHandler;

    ShapeMode mShapeMode = ShapeMode::Sphere;
    GravityCenter mGravityCenter = { make_float3(0.0f, 0.0f, 0.0f), 0.06f, GravityMode::Off };
    bool mGravityFollow = false;
    size_t mParticleCount = 1000000;
};