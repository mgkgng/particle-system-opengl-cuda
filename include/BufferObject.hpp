#pragma once

#include "gl_common.hpp"

class BufferObject {
public:
    BufferObject(GLenum target, GLenum usage) : mTarget(target), mUsage(usage) { glGenBuffers(1, &mID); }
    ~BufferObject() { glDeleteBuffers(1, &mID); }

    void Bind() { glBindBuffer(mTarget, mID); }
    void Unbind() { glBindBuffer(mTarget, 0); }

    void InitializeData(const void* ptr, size_t size) {
        glBindBuffer(mTarget, mID);
        glBufferData(mTarget, size, ptr, mUsage);
    }
    
private:
    GLenum mTarget;
    GLenum mUsage;
    GLuint mID;
};