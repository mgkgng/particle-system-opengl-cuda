#version 430 core

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

struct Particle {
    vec3 position;
    vec3 velocity;
    vec4 color;
    float life;
};

layout(std430, binding = 0) buffer ParticleData {
    Particle particles[];
};

void main() {
    uint id = gl_GlobalInvocationID.x;
    particles[id].position += vec3(0.0, 0.2, 0.0);
}