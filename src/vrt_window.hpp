#ifndef __VULKAN_RAY_TRACING_WINDOW_HPP__
#define __VULKAN_RAY_TRACING_WINDOW_HPP__

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace vrt {
	class Window {
	public:
		Window();
		~Window();

		void createWindowSurface(VkInstance instance, VkSurfaceKHR* surface);
		void getFrameBufferSize(int* width, int* height);

		bool isMinimized();
		bool shouldClose();

		GLFWwindow* getWindowHandle() const { return _window; }

	private:
		GLFWwindow* _window;
	};
}

#endif