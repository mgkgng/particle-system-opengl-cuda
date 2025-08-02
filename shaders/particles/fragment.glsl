#version 430 core

uniform vec3 uLightPos;

in vec3 vWorldPos;
in vec4 vParticleColor;
in vec4 vLightColor;
out vec4 FragColor;

void main() {
    if (uLightPos.z != 0.0f) {
        FragColor = vParticleColor;
        return;
    }

    float dist = length(uLightPos - vWorldPos);

    float lightRadius = 6.5;
    // float intensity = clamp(1.0 - (dist / lightRadius), 0.0, 1.0);
    float intensity = pow(1.0 - smoothstep(0.0, lightRadius, dist), 2.0);

    vec4 finalColor = mix(vParticleColor, vLightColor, intensity);
    FragColor = finalColor;
}