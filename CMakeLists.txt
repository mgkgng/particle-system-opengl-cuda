cmake_minimum_required(VERSION 3.15)
project(ParticleSystem LANGUAGES CXX CUDA)

# cpp configuration
set(CMAKE_CXX_STANDARD 20)

# cuda configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Avoid building in source directory
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Please create a separate 'build/' directory.")
endif()

# Platform-specific executable name
set(EXECUTABLE_NAME "particle_system")

# Disable unnecessary GLFW/GLM targets
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE OFF CACHE BOOL "" FORCE)

# Add third-party dependencies
add_subdirectory(externals/glfw)
add_subdirectory(externals/glm)

# Source files
file(GLOB SRC_FILES "src/*.cpp")
list(FILTER SRC_FILES EXCLUDE REGEX "glad\\.c$")
set(GLAD_SRC src/glad/glad.c)

file(GLOB CUDA_SOURCES "assets/kernels/*.cu")
list(APPEND SRC_FILES ${CUDA_SOURCES})

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
        cuda
        cudart
    )
elseif(UNIX AND NOT APPLE)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE
        glfw
        glm::glm
        ${CMAKE_DL_LIBS}
        pthread
        m
        GL
        cuda
        cudart
    )
elseif(APPLE)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE
        glfw
        glm::glm
        "-framework OpenGL"
        "-framework Cocoa"
        "-framework IOKit"
        cuda
        cudart
    )
endif()
