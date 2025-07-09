#include "utils.hpp"

std::string loadFileSource(const std::string& path) {
    std::ifstream file;
    std::stringstream ss;

    file.open(path);
    if (!file.is_open()) {
        std::cout << "Cannot open the shader file." << std::endl;
        return "";
    }
    ss << file.rdbuf();
    file.close();
    return ss.str();
}
