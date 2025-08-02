#pragma once

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include "Window.hpp"
#include "ParticleSystem.hpp"
#include "ProgramConfig.hpp"

class ImGuiLayer {
public:
    ImGuiLayer(GLFWwindow* window, ProgramConfig* programConfig);
    ~ImGuiLayer();

    void BeginFrame();
    void SetUI();
    void Draw(ParticleSystem& system);
    void EndFrame();

    bool IsHovered() { return ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow); }

private:
    ProgramConfig* mProgramConfig;
};