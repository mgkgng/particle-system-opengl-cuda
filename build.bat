@echo off
setlocal ENABLEDELAYEDEXPANSION

REM === Parse argument ===
set MODE=%1

if "%MODE%"=="" (
    set MODE=full
) else if /I "%MODE%"=="full" (
    set BUILD_MODE=full
) else if /I "%MODE%"=="re" (
    set BUILD_MODE=re
) else (
    echo ❌ Error: Unknown argument "%MODE%"
    echo Usage:
    echo    build.bat           - Full build
    echo    build.bat full      - Full build
    echo    build.bat re        - Fast incremental rebuild
    exit /b 1
)

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

REM === Clean cache only if full build ===
if /I "%BUILD_MODE%"=="full" (
    echo 🧹 Cleaning CMake cache...
    del /q CMakeCache.txt >nul 2>&1
    rmdir /s /q CMakeFiles >nul 2>&1
    cmake .. -A x64
)

REM === Build ===
cmake --build . --config Release

echo.
echo Build done (%BUILD_MODE% mode).
echo.
