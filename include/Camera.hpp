#pragma once

#include "gl_common.hpp"

class Camera {
public:
    static constexpr float kFOV = 45.0f;
    static constexpr float kNearPlane = 1.0f;
    static constexpr float kFarPlane = 100.0f;

    static constexpr float kRotSensitivity = 0.05f;
    static constexpr float kZoomSpeed = 1.05f;
    static constexpr float kMinDistanceToOrigin = 0.1f;
    static constexpr float kMaxDistanceToOrigin = 100.0f;

    static constexpr float kInitialPosX = 0.0f;
    static constexpr float kInitialPosY = 0.0f;
    static constexpr float kInitialPosZ = -15.0f;

    Camera();

    void Translate(float deltaTime);
    void Rotate(float yaw, float pitch);
    void Zoom(const float dz);

    glm::mat4 GetViewProjMatrix() const;

    bool IsUpdated() const { return mIsUpdated; }
    void SetUpdated(bool updated) { mIsUpdated = updated; }

private:
    void UpdateView();

    glm::vec3 mPosition = glm::vec3(kInitialPosX, kInitialPosY, kInitialPosZ);
    glm::vec3 mUp = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::mat4 mViewMatrix = glm::mat4(1.0f);
    glm::mat4 mProjMatrix = glm::mat4(1.0f);

    float mYaw = -90.0f;
    float mPitch = 0.0f;
    float mDistance = glm::distance(mPosition, mUp);

    bool mIsUpdated = true;
};

class CameraController {
public:
    CameraController(Camera* camera);
    
    void Update(float deltaTime);
    void OnKeyInput(int key, int action);
    void SetMovementSpeed(float maxSpeed);
    void SetAcceleration(float a);

private:
    Camera *mCamera;

    glm::vec3 mVelocity;
    glm::vec3 mAcceleration;
    glm::vec3 mTargetAcceleration;

    float mMaxSpeed;

};