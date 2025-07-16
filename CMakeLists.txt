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

# Define the executable
add_executable(${EXECUTABLE_NAME}
    ${SRC_FILES}
    ${GLAD_SRC}
)

# Include directories
target_include_directories(${EXECUTABLE_NAME} PRIVATE
    include
    src/glad
    externals/glfw/include
    externals/glm
)

# Platform-specific linking
if(WIN32)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE
        glfw
        glm::glm
        opengl32
    )
elseif(UNIX AND NOT APPLE)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE
        glfw
        glm::glm
        ${CMAKE_DL_LIBS}
        pthread
        m
        GL
    )
elseif(APPLE)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE
        glfw
        glm::glm
        "-framework OpenGL"
        "-framework Cocoa"
        "-framework IOKit"
    )
endif()
