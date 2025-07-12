#pragma once

#include "gl_common.hpp"
#include "utils.hpp"

class Shader {
public:
    Shader(const std::string& path);
    ~Shader() { glDeleteProgram(mID); }

    void Use() const { glUseProgram(mID); }
    void Unuse() const { glUseProgram(0); }

    // void SetUniform(const std::string& name, int value) const;
    // void SetUniform(const std::string& name, float value) const;
    // void SetUniform(const std::string& name, const glm::vec3& vec) const;
    // void SetUniform(const std::string& name, const glm::mat4& mat) const;

    static void CheckCompileError(GLuint id, const std::string& type);

protected:
    Shader() = default;

    GLuint mID;
};

class ComputeShader : public Shader {
public:
    ComputeShader(const std::string& name);
    
    void Compute(size_t x, size_t y, size_t z) { glDispatchCompute(x, y, z); }

    void SetMemoryBarrier() { glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT); }
};