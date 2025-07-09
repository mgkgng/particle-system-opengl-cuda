#include "Shader.hpp"

const std::string SHADER_PATH = "assets/shaders/";

Shader::Shader(const std::string& name) {
    const std::string vertexStr = loadFileSource(SHADER_PATH + name + "/vertex.glsl");
    const std::string fragmentStr = loadFileSource(SHADER_PATH + name + "/fragment.glsl");

    if (vertexStr.empty() || fragmentStr.empty()) {
        throw std::runtime_error("Error: Empty shader source");
    }

    const char *vertexSrc = vertexStr.c_str();
    const char *fragmentSrc = fragmentStr.c_str();

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSrc, NULL);
    glCompileShader(vertexShader);
    CheckCompileError(vertexShader, GL_SHADER);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSrc, NULL);
    glCompileShader(fragmentShader);
    CheckCompileError(fragmentShader, GL_SHADER);

    mID = glCreateProgram();
    glAttachShader(mID, vertexShader);
    glAttachShader(mID, fragmentShader);
    glLinkProgram(mID);
    CheckCompileError(mID, GL_PROGRAM);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void Shader::CheckCompileError(GLuint id, int type) {
    int success;
    char infoLog[1024];

    if (type == GL_SHADER) {
        glGetShaderiv(id, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(id, 1024, NULL, infoLog);
            std::cerr << "Error while compiling shader:\n" << infoLog << std::endl;
            exit(EXIT_FAILURE);
        }
    } else {
        glGetProgramiv(id, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(id, 1024, NULL, infoLog);
            std::cerr << "Error while linking program:\n" << infoLog << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}
