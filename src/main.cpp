#include "vrt_ray_tracer.hpp"
#include "vrt_camera.hpp"

#include <iostream>
#include <chrono>

int main() {
    vrt::Window window{};
    vrt::RayTracer rayTracer{ window };

    vrt::Camera camera{ 40.0f, 1024.0f / 768.0f };

    glm::vec3 lightDirection{ 1.0f, -2.0f, 0.5f };
    lightDirection = glm::normalize(lightDirection);

    vrt::Settings settings{};
    settings.projection = camera.getProjectionMatrix();
    settings.skyColor = { 0.53f, 0.81f, 0.92f };
    settings.directionalLight = { lightDirection, 1.0f };

    float lightAngle = 10.0f;

    std::cout << "Init done!" << std::endl;

    auto currentTime = std::chrono::high_resolution_clock::now();

    while (!window.shouldClose()) {
        glfwPollEvents();

        auto newTime = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float, std::chrono::seconds::period>(newTime - currentTime).count();

        camera.move(window.getWindowHandle(), elapsed);
        settings.transform = camera.getWorldTransform();
        settings.angle += elapsed * 0.8f;

        if (!window.isMinimized()) {
            rayTracer.updateSettings(settings);
            rayTracer.drawFrame();
        }

        currentTime = newTime;
    }

    return 0;
}