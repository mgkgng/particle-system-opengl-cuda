#pragma once

#include <string>
#include <iostream>

enum class ShapeMode { Cube, Sphere };
enum class GravityMode { Off, Static, Follow };

class ProgramConfig {

public:
    bool ParseArg(int argc, char **argv);

private:
    friend class Application;

    ShapeMode mShapeMode = ShapeMode::Sphere;
    GravityMode mGravityMode = GravityMode::Off;
};