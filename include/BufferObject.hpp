#pragma once

#include <stdexcept>

#include "gl_common.hpp"

class BufferObject {
public:
    BufferObject(GLenum target) : mTarget(target) { glGenBuffers(1, &mID); }
    BufferObject(GLenum target, GLenum usage) : mTarget(target), mUsage(usage) { glGenBuffers(1, &mID); }
    ~BufferObject() { glDeleteBuffers(1, &mID); }

    BufferObject(const BufferObject& other) = delete;
    BufferObject& operator=(const BufferObject& other) = delete;

    void Bind() { glBindBuffer(mTarget, mID); }
    void Unbind() { glBindBuffer(mTarget, 0); }

    void InitializeData(const void* ptr, size_t size) {
        glBindBuffer(mTarget, mID);
        glBufferData(mTarget, size, ptr, mUsage);
    }

    void BindIndexedTarget(size_t index) {
        if (mTarget != GL_ATOMIC_COUNTER_BUFFER && mTarget != GL_TRANSFORM_FEEDBACK_BUFFER && mTarget != GL_UNIFORM_BUFFER && mTarget != GL_SHADER_STORAGE_BUFFER) {
            std::runtime_error("Wrong buffer target for the bind operation.");
        }

        glBindBufferBase(mTarget, index, mID);
    }

    void* MapBuffer(GLenum access) {
        glBindBuffer(mTarget, mID);
        return glMapBuffer(mTarget, access);
    }

    void UnmapBuffer() { glUnmapBuffer(mTarget); }

    GLuint GetID() const { return mID; }
    
private:
    GLenum mTarget;
    GLenum mUsage = GL_STATIC_DRAW;
    GLuint mID;
};