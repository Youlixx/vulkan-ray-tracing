#include "vrt_window.hpp"

#include <stdexcept>

namespace vrt {
	Window::Window() {
		if (glfwInit() != GLFW_TRUE) {
			throw std::runtime_error("Unable to initialize GLFW");
		}

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		_window = glfwCreateWindow(1024, 768, "Vulkan Ray Tracing", nullptr, nullptr);
	}

	Window::~Window() {
		glfwDestroyWindow(_window);
		glfwTerminate();
	}

	void Window::createWindowSurface(VkInstance instance, VkSurfaceKHR* surface) {
		if (glfwCreateWindowSurface(instance, _window, nullptr, surface) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create surface");
		}
	}

	void Window::getFrameBufferSize(int* width, int* height) {
		glfwGetFramebufferSize(_window, width, height);
	}

	bool Window::isMinimized() {
		return glfwGetWindowAttrib(_window, GLFW_ICONIFIED) == VK_TRUE;
	}

	bool Window::shouldClose() {
		return glfwWindowShouldClose(_window);
	}
}