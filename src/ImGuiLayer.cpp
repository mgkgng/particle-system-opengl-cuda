#include "ImGuiLayer.hpp"

ImGuiLayer::ImGuiLayer(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // ImGui_ImplGlfw_InitForOpenGL(window, true);
    // ImGui_ImplOpenGL3_Init("#version 330");
}

ImGuiLayer::~ImGuiLayer() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void ImGuiLayer::SetUI() {
    ImGui::Begin("Controls");
    // ImGui::SliderFloat("Size", &particleSize, 0.1f, 10.0f);
    // ImGui::ColorEdit3("Color", (float*)&particleColor);
    // ImGui::Checkbox("Enable Gravity", &gravityEnabled);
    ImGui::End();
}

void ImGuiLayer::BeginFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayer::Draw(ParticleSystem& system) {
    ImGui::Begin("Particle Settings");
    // Example controls
    // ImGui::SliderFloat("Emission Rate", &system.GetEmissionRate(), 0.0f, 100.0f);
    // ImGui::ColorEdit3("Particle Color", (float*)&system.GetColor());
    ImGui::End();
}

void ImGuiLayer::EndFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}