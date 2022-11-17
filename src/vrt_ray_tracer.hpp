#ifndef __VULKAN_RAY_TRACING_RAY_TRACER_HPP__
#define __VULKAN_RAY_TRACING_RAY_TRACER_HPP__

#include "vrt_window.hpp"

#include <glm/glm.hpp>

#include <vector>

namespace vrt {
	struct Settings {
		alignas(16) glm::mat4 projection;
		alignas(16) glm::mat4 transform;
		alignas(16) glm::vec4 directionalLight;

		alignas(16) float angle;
	};

	struct Sphere {
		glm::vec3 position;
		float radius;
		glm::vec3 albedo;
		alignas(16) glm::vec3 specular;
	};

	struct Plane {
		alignas(16) glm::vec3 position;
		alignas(16) glm::vec3 normal;
		alignas(16) glm::vec3 albedo;
		alignas(16) glm::vec3 specular;
	};

	class RayTracer {
	public:
		RayTracer(Window& window);
		~RayTracer();

		RayTracer(RayTracer&) = delete;
		RayTracer& operator=(RayTracer&) = delete;

		void drawFrame();
		void updateSettings(Settings& settings);

	private:
		void createInstance();
		void createDevice();
		void createCommandPools();
		void createSwapChain();
		void createTargetTexture();
		void createSkyBox();
		void createStorageBuffers();
		void createDescriptorSets();
		void createGraphicsPipeline();
		void createComputePipeline();
		void createDrawCommandBuffers();
		void createComputeCommandBuffer();
		void createSemaphoresAndFences();

		uint8_t getPhysicalDeviceQuality(VkPhysicalDevice physicalDevice);

		static bool getGraphicsQueueFamilyIndex(std::vector<VkQueueFamilyProperties>& queueFamilyProperties, uint32_t* queueFamilyIndex);
		static bool getComputeQueueFamilyIndex(std::vector<VkQueueFamilyProperties>& queueFamilyProperties, uint32_t* queueFamilyIndex);
		static bool getTransferQueueFamilyIndex(std::vector<VkQueueFamilyProperties>& queueFamilyProperties, uint32_t* queueFamilyIndex);

		VkSurfaceFormatKHR selectSurfaceFormat();
		VkPresentModeKHR selectPresentMode();
		VkSurfaceCapabilitiesKHR getSurfaceCapabilities();

		uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

		void createCommandBuffers(VkCommandPool commandPool, VkCommandBuffer* commandBuffers, uint32_t commandBufferCount = 1);
		void submitCommandBuffers(VkCommandPool commandPool, VkQueue queue, VkCommandBuffer* commandBuffers, uint32_t commandBufferCount = 1);

		void createBuffer(VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& memory);
		void createImageAndView(VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& memory, VkImageView& view, uint32_t width, uint32_t height);
		void createCubeMap(VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& memory, VkImageView& view, uint32_t width, uint32_t height);
		void changeImageLayout(VkImageLayout oldLayout, VkImageLayout newLayout, VkImage image, VkAccessFlags srcAccessMask = 0, VkAccessFlags dstAccessMask = 0, uint32_t layerCount = 1);

		void createStorageBuffer(VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& bufferMemory, void* data);

		void loadShaderModule(const char* path, VkShaderModule& shaderModule);

	private:
		const std::vector<const char*> REQUIRED_EXTENSION_PROPERTIES{
			VK_KHR_SWAPCHAIN_EXTENSION_NAME
		};

		const std::vector<const char*> REQUIRED_LAYERS{
			"VK_LAYER_KHRONOS_validation"
		};

		static const char* SHADER_VERTEX_PATH;
		static const char* SHADER_FRAGMENT_PATH;
		static const char* SHADER_COMPUTE_PATH;

		static const char* SKY_BOX_TEXTURE_PATHS[6];

	private:
		Window& _window;

		VkInstance _instance;
		VkSurfaceKHR _surface;
		VkDevice _logicalDevice;
		VkPhysicalDevice _physicalDevice;

		VkDescriptorPool _descriptorPool;
		VkSampler _sampler;

		struct {
			uint32_t graphics;
			uint32_t compute;
			uint32_t transfer;
		} _queueFamilyIndices;

		struct {
			VkSwapchainKHR swapChain;
			VkFormat format;
			VkExtent2D extent;

			std::vector<VkImage> images;
			std::vector<VkImageView> imageViews;
			std::vector<VkFramebuffer> frameBuffers;

			VkRenderPass renderPass;

			uint32_t imageCount;
		} _swapChain;

		struct {
			VkDescriptorSetLayout descriptorSetLayout;
			VkDescriptorSet descriptorSet;
			
			VkCommandPool commandPool;
			VkQueue queue;

			VkPipeline pipeline;
			VkPipelineLayout pipelineLayout;

			std::vector<VkCommandBuffer> drawCommandBuffers;
		} _graphics;

		struct {
			VkDescriptorSetLayout descriptorSetLayout;
			VkDescriptorSet descriptorSet;

			VkCommandPool commandPool;
			VkQueue queue;

			VkPipeline pipeline;
			VkPipelineLayout pipelineLayout;

			VkCommandBuffer commandBuffer;
		} _compute;

		struct {
			VkImage image;
			VkImageView imageView;
			VkDeviceMemory imageDeviceMemory;
		} _targetTexture;

		struct {
			VkImage image;
			VkImageView imageView;
			VkDeviceMemory imageDeviceMemory;
		} _skyBox;

		struct {
			VkBuffer sphereBuffer;
			VkDeviceMemory sphereMemory;

			VkBuffer planeBuffer;
			VkDeviceMemory planeMemory;

			Settings settings;
			VkBuffer settingBuffer;
			VkDeviceMemory settingMemory;
			void* settingHandle;
		} _scene;

		struct {
			VkFence computeComplete;
			VkSemaphore presentComplete;
			VkSemaphore renderComplete;
		} _sync;
	};
}

#endif