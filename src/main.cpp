#include "Application.hpp"
#include "VertexArrayObject.hpp"

int main() {
    Application app(800, 800, "Particle Systems");

    if (!app.InitCUDA())
        return EXIT_FAILURE;

    app.Run();
}
