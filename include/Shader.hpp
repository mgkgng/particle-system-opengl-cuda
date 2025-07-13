#pragma once

#include "gl_common.hpp"
#include "utils.hpp"

class Shader {
public:
    Shader(const std::string& path);
    ~Shader() { glDeleteProgram(mID); }

    void Use() const { glUseProgram(mID); }
    void Unuse() const { glUseProgram(0); }

    void SetUniform(const std::string& name, int value) { glUniform1i(GetUniformLocation(name), value); }
    void SetUniform(const std::string& name, float value) { glUniform1f(GetUniformLocation(name), value); }
    void SetUniform(const std::string& name, const glm::vec3& vec) { glUniform3fv(GetUniformLocation(name), 1, glm::value_ptr(vec)); }
    void SetUniform(const std::string& name, const glm::mat4& mat) { glUniformMatrix4fv(GetUniformLocation(name), 1, GL_FALSE, glm::value_ptr(mat)); }

    static void CheckCompileError(GLuint id, const std::string& type);

protected:
    Shader() = default;

    GLint GetUniformLocation(const std::string& name);

    GLuint mID;
    std::unordered_map<std::string, GLint> mUniformLocations;
};

class ComputeShader : public Shader {
public:
    ComputeShader(const std::string& name);
    
    void Compute(size_t x, size_t y, size_t z) { glDispatchCompute(x, y, z); }
};