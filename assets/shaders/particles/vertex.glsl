#version 430 core

struct Particle {
    vec3 position;
    vec3 velocity;
    vec4 color;
    float life;
};

layout(std430, binding = 0) buffer ParticleData {
    Particle particles[];
};

out vec4 vColor;

void main() {
    Particle p = particles[gl_VertexID];
    vColor = p.color;
    gl_Position = vec4(p.position, 1.0);
}