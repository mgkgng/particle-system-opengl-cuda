#version 430 core

struct Particle {
    vec4 position;
    vec4 velocity;
    vec4 color;
    float lifespan;
    float _pad1, _pad2, _pad3;
};

layout(std430, binding = 0) buffer ParticleData {
    Particle particles[];
};

out vec4 vColor;

void main() {
    Particle p = particles[gl_VertexID];
    vColor = p.color;
    gl_Position = p.position;
}