#version 430 core

struct Particle {
    vec3 position;
    float _pad1;

    vec3 _initialPosition;
    float _pad2;

    vec3 _velocity;
    float _pad3;

    vec4 particleColor;
    vec4 lightColor;

    float lifespan;
    float size;
    float _pad4, _pad5;
};

layout(std430, binding = 0) buffer ParticleData {
    Particle particles[];
};

uniform mat4 uProjView;
out vec4 vParticleColor;
out vec4 vLightColor;
out vec3 vWorldPos;

void main() {
    Particle p = particles[gl_VertexID];

    vParticleColor = p.particleColor;
    vLightColor = p.lightColor;
    vWorldPos = vec3(p.position.x, p.position.y, p.position.z);

    gl_Position = uProjView * vec4(p.position, 1.0f);
    gl_PointSize = 2.0f;
}