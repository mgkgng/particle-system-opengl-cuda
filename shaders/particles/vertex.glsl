#version 430 core

struct Particle {
    vec4 position;
    vec4 initialPosition;
    vec4 velocity;
    vec4 color;
    float lifespan;
    float _pad1, _pad2;
};

layout(std430, binding = 0) buffer ParticleData {
    Particle particles[];
};

uniform mat4 uProjView;
out vec4 vColor;

void main() {
    Particle p = particles[gl_VertexID];
    vColor = p.color;
    gl_Position = uProjView * p.position;
}