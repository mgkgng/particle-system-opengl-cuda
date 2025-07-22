#pragma once

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include "Window.hpp"
#include "ParticleSystem.hpp"

class ImGuiLayer {
public:
    ImGuiLayer(GLFWwindow* window);
    ~ImGuiLayer();

    void BeginFrame();
    void SetUI();
    void Draw(ParticleSystem& system);
    void EndFrame();
};