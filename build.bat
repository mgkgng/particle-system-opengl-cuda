@echo off
setlocal ENABLEDELAYEDEXPANSION

REM === Locate Visual Studio with vswhere ===
set VSWHERE="%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`%VSWHERE% -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set VSINSTALL=%%i
)

if not defined VSINSTALL (
    echo ❌ Visual Studio not found. Please install it with C++ toolset.
    exit /b 1
)

REM === Call vcvars64.bat to set environment ===
call "!VSINSTALL!\VC\Auxiliary\Build\vcvars64.bat"

REM === Init submodules ===
git submodule update --init --recursive

REM === Create and enter build directory ===
if not exist build (
    mkdir build
)
cd build

REM === Clean up cache if needed ===
del /q CMakeCache.txt >nul 2>&1
rmdir /s /q CMakeFiles >nul 2>&1

REM === Configure (Visual Studio generator auto-detected from vcvars) ===
cmake .. -A x64

REM === Build with CMake ===
cmake --build . --config Release

REM === Run executable ===
if exist ".\Release\particle_system.exe" (
    .\Release\particle_system.exe
) else (
    echo ❌ Executable not found. Build may have failed.
)
