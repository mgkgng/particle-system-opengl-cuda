#include "ProgramConfig.hpp"
#include "Application.hpp"

int main(int argc, char **argv) {
    ProgramConfig programConfig;
    if (!programConfig.ParseArg(argc, argv)) return EXIT_FAILURE;

    Application app(programConfig);
    if (!app.InitCUDA()) return EXIT_FAILURE;

    app.Run();
}
