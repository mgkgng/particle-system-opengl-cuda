#!/bin/bash
set -e

# === Initialize submodules ===
git submodule update --init --recursive

# === Create build directory if not exist ===
mkdir -p build
cd build

# === Run CMake ===
cmake ..
cmake --build .

# === Run the executable ===
./particle_system
