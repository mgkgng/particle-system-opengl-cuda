#include "Shader.hpp"

const std::string SHADER_PATH = "assets/shaders/";
const std::string KERNEL_PATH = "assets/kernels/";

Shader::Shader(const std::string& name) {
    const std::string vertexStr = LoadShaderSource(SHADER_PATH + name + "/vertex.glsl");
    const std::string fragmentStr = LoadShaderSource(SHADER_PATH + name + "/fragment.glsl");

    if (vertexStr.empty() || fragmentStr.empty()) {
        throw std::runtime_error("Error: Empty shader source");
    }

    const char *vertexSrc = vertexStr.c_str();
    const char *fragmentSrc = fragmentStr.c_str();

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSrc, NULL);
    glCompileShader(vertexShader);
    CheckCompileError(vertexShader, "SHADER");

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSrc, NULL);
    glCompileShader(fragmentShader);
    CheckCompileError(fragmentShader, "SHADER");

    mID = glCreateProgram();
    glAttachShader(mID, vertexShader);
    glAttachShader(mID, fragmentShader);
    glLinkProgram(mID);
    CheckCompileError(mID, "PROGRAM");

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

GLint Shader::GetUniformLocation(const std::string& name) {
    auto it = mUniformLocations.find(name);
    if (it != mUniformLocations.end()) return it->second;

    GLint location = glGetUniformLocation(mID, name.c_str());
    if (location == -1) {
        std::cerr << "Warning: Uniform '" << name << "' cannot be found." << std::endl; 
    }
    mUniformLocations[name] = location;
    return location;
}

void Shader::CheckCompileError(GLuint id, const std::string& type) {
    int success;
    char infoLog[1024];

    if (type == "SHADER") {
        glGetShaderiv(id, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(id, 1024, NULL, infoLog);
            std::cerr << "Error while compiling shader:\n" << infoLog << std::endl;
            exit(EXIT_FAILURE);
        }
    } else if (type == "PROGRAM") {
        glGetProgramiv(id, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(id, 1024, NULL, infoLog);
            std::cerr << "Error while linking program:\n" << infoLog << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

std::string Shader::LoadShaderSource(const std::string &path) {
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
