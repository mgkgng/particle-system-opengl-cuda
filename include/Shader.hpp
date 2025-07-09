#pragma once

#include "gl_common.hpp"
#include "utils.hpp"

#define GL_SHADER 0
#define GL_PROGRAM 1

class Shader {
public:
    Shader(const std::string& path);
    ~Shader() { glDeleteProgram(mID); }

    void Bind() const { glUseProgram(mID); }
    void Unbind() const { glUseProgram(0); }

    // void SetUniform(const std::string& name, int value) const;
    // void SetUniform(const std::string& name, float value) const;
    // void SetUniform(const std::string& name, const glm::vec3& vec) const;
    // void SetUniform(const std::string& name, const glm::mat4& mat) const;

    static void CheckCompileError(GLuint id, int type);
private:
    GLuint mID;

};