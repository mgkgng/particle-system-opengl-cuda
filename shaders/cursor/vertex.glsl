#version 430 core

uniform vec2 uCursorPos;

void main() {
    gl_Position = vec4(uCursorPos, 0.0, 1.0);
    gl_PointSize = 3.0;
}
