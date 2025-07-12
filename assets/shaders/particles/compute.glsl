#version 430 core

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

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

void main() {
    uint id = gl_GlobalInvocationID.x;
    particles[id].position += vec4(0.0f, 0.2f, 0.0f, 0.0f);
}