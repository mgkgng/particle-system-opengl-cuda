#pragma once

#include "gl_common.hpp"

class VertexArrayObject {
public:
    VertexArrayObject() { glGenVertexArrays(1, &mID); }
    ~VertexArrayObject() { glDeleteVertexArrays(1, &mID); }

    void Bind() { glBindVertexArray(mID); }
    void Unbind() { glBindVertexArray(0); }

    void SetAttribute(size_t index, size_t count, GLenum type, bool normalized, size_t stride, const void* offset) {
        glVertexAttribPointer(index, count, type, normalized, stride, offset);
        glEnableVertexAttribArray(index);
    }
private:
    GLuint mID;
};