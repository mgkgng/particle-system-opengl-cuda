#include "ProgramConfig.hpp"

bool ProgramConfig::ParseArg(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--shape" && i + 1 < argc) {
            std::string value = argv[++i];
            if (value == "sphere") mShapeMode = ShapeMode::Sphere;
            else if (value == "cube") mShapeMode = ShapeMode::Cube;
            else {
                std::cerr << "Unknown shape: " << value << " (--help to see help)." << std::endl;
                return false;
            }

        } else if (arg == "--gravity" && i + 1 < argc) {
            std::string value = argv[++i];
            if (value == "off") mGravityCenter.mode = GravityMode::Off;
            else if (value == "static") {
                mGravityCenter.mode = GravityMode::Static;
                mGravityFollow = false;
            } else if (value == "follow") {
                mGravityCenter.mode = GravityMode::Follow;
                mGravityFollow = true;
            } else {
                std::cerr << "Unknown gravity mode: " << value << " (--help to see help)." << std::endl;
                return false;
            }
        } else if (arg == "--count") {
            std::string value = argv[++i];
            try {
                int count = std::stoi(value);
                if (count <= 0) {
                    std::cerr << "Particle count must be a positive integer." << std::endl;
                    return false;
                } else if (count >= 3000000) {
                    std::cerr << "Maximum number of particles is 3000000." << std::endl;
                    return false;
                }
                mParticleCount = count;
            } catch (const std::exception& e) {
                std::cerr << "Invalid number for particle count: " << value << std::endl;
                return false;
            }

        } else if (arg == "--help") {
            std::cout << "Usage: ParticleSystem [options]\n"
                    << "Options:\n"
                    << "  --help                         Show this help message\n"
                    << "  --shape <sphere|cube>          Shape to initialize particles (default: sphere)\n"
                    << "  --gravity <off|static|follow>  Gravity behavior (default: off)\n"
                    << std::endl;
            return false;
        } else {
            std::cerr << "[Error] Wrong arguments (--help to see help)." << std::endl;
            return false;
        }
    }
    return true;
}