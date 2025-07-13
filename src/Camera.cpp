#include "Camera.hpp"

Camera::Camera(float fov, float aspectRatio, float nearPlane, float farPlane) {
    mViewMatrix = glm::lookAt(mPosition, glm::vec3(0.0f), mUp);
    mProjMatrix = glm::perspective(glm::radians(fov), aspectRatio, nearPlane, farPlane);
}

void Camera::Translate(float deltaTime) {

    UpdateView();
}

void Camera::Rotate(float deltaYaw, float deltaPitch) {
    mYaw += deltaYaw * Camera::kRotSensitivity;
    mPitch += deltaPitch * Camera::kRotSensitivity;
    mPitch = glm::clamp(mPitch, -89.0f, 89.0f);

    UpdateView();
}

void Camera::Zoom(float dz) {
    mDistance *= (1.0f + dz * Camera::kZoomSpeed);
    mDistance = glm::clamp(mDistance, Camera::kMinDistanceToOrigin, kMaxDistanceToOrigin);
    UpdateView();
}

glm::mat4 Camera::GetViewProjMatrix() const {
    return mProjMatrix * mViewMatrix;
}

void Camera::UpdateView() {
    float yawRad = glm::radians(mYaw);
    float pitchRad = glm::radians(mPitch);

    glm::vec3 direction;
    direction.x = cos(pitchRad) * sin(yawRad);
    direction.y = sin(pitchRad);
    direction.z = cos(pitchRad) * cos(yawRad);
    direction = glm::normalize(direction);

    mPosition = glm::vec3(0.0f) - direction * mDistance;

    glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), direction));
    mUp = glm::normalize(glm::cross(direction, right));

    mViewMatrix = glm::lookAt(mPosition, glm::vec3(0.0f), mUp);
    mIsUpdated = true;
}


