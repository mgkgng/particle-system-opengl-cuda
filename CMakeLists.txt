cmake_minimum_required(VERSION 3.15)
project(ParticleSystem)

set(CMAKE_CXX_STANDARD 20)

# GLFW config
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# GLM config (disable test building)
set(GLM_TEST_ENABLE OFF CACHE BOOL "" FORCE)

# Add third-party dependencies
add_subdirectory(externals/glfw)
add_subdirectory(externals/glm)

# Source files
file(GLOB SRC_FILES "src/*.cpp")
list(FILTER SRC_FILES EXCLUDE REGEX "glad\\.c$")
set(GLAD_SRC src/glad/glad.c)

add_executable(particle_system
    ${SRC_FILES}
    ${GLAD_SRC}
)

# Includes
target_include_directories(particle_system PRIVATE
    include
    src/glad
    externals/glfw/include
    externals/glm        # for GLM headers
)

# Link libraries
target_link_libraries(particle_system PRIVATE
    glfw
    glm::glm
    dl
    pthread
    m
    GL
)
