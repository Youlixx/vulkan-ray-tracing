#include "vrt_ray_tracer.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb.h>

#include <glm/gtx/transform.hpp>

#include <stdexcept>
#include <fstream>
#include <set>

namespace vrt {
	const char* RayTracer::SHADER_VERTEX_PATH = "data/shaders/rendering.vert.spv";
	const char* RayTracer::SHADER_FRAGMENT_PATH = "data/shaders/rendering.frag.spv";
	const char* RayTracer::SHADER_COMPUTE_PATH = "data/shaders/ray_tracing.comp.spv";

	const char* RayTracer::SKY_BOX_TEXTURE_PATHS[6] = {
		"data/skybox/back.jpg",
		"data/skybox/front.jpg",
		"data/skybox/top.jpg",
		"data/skybox/bottom.jpg",
		"data/skybox/right.jpg",
		"data/skybox/left.jpg"
	};

	RayTracer::RayTracer(Window& window) : _window{ window } {
		createInstance();
		createDevice();
		createCommandPools();
		createSwapChain();
		createTargetTexture();
		createSkyBox();
		createStorageBuffers();
		createDescriptorSets();
		createGraphicsPipeline();
		createComputePipeline();
		createDrawCommandBuffers();
		createComputeCommandBuffer();
		createSemaphoresAndFences();
	}

	RayTracer::~RayTracer() {
		vkDeviceWaitIdle(_logicalDevice);

		vkDestroyFence(_logicalDevice, _sync.computeComplete, nullptr);
		vkDestroySemaphore(_logicalDevice, _sync.presentComplete, nullptr);
		vkDestroySemaphore(_logicalDevice, _sync.renderComplete, nullptr);

		vkDestroyPipeline(_logicalDevice, _compute.pipeline, nullptr);
		vkDestroyPipelineLayout(_logicalDevice, _compute.pipelineLayout, nullptr);
		vkDestroyPipeline(_logicalDevice, _graphics.pipeline, nullptr);
		vkDestroyPipelineLayout(_logicalDevice, _graphics.pipelineLayout, nullptr);

		vkFreeMemory(_logicalDevice, _scene.planeMemory, nullptr);
		vkDestroyBuffer(_logicalDevice, _scene.planeBuffer, nullptr);
		vkFreeMemory(_logicalDevice, _scene.sphereMemory, nullptr);
		vkDestroyBuffer(_logicalDevice, _scene.sphereBuffer, nullptr);
		vkUnmapMemory(_logicalDevice, _scene.settingMemory);
		vkFreeMemory(_logicalDevice, _scene.settingMemory, nullptr);
		vkDestroyBuffer(_logicalDevice, _scene.settingBuffer, nullptr);

		vkDestroyImageView(_logicalDevice, _skyBox.imageView, nullptr);
		vkDestroyImage(_logicalDevice, _skyBox.image, nullptr);
		vkFreeMemory(_logicalDevice, _skyBox.imageDeviceMemory, nullptr);

		vkDestroyImageView(_logicalDevice, _targetTexture.imageView, nullptr);
		vkDestroyImage(_logicalDevice, _targetTexture.image, nullptr);
		vkFreeMemory(_logicalDevice, _targetTexture.imageDeviceMemory, nullptr);

		vkDestroySampler(_logicalDevice, _sampler, nullptr);

		vkDestroyDescriptorSetLayout(_logicalDevice, _compute.descriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(_logicalDevice, _graphics.descriptorSetLayout, nullptr);
		vkDestroyDescriptorPool(_logicalDevice, _descriptorPool, nullptr);

		for (auto frameBuffer : _swapChain.frameBuffers) {
			vkDestroyFramebuffer(_logicalDevice, frameBuffer, nullptr);
		}

		vkDestroyRenderPass(_logicalDevice, _swapChain.renderPass, nullptr);

		for (auto imageView : _swapChain.imageViews) {
			vkDestroyImageView(_logicalDevice, imageView, nullptr);
		}

		vkDestroySwapchainKHR(_logicalDevice, _swapChain.swapChain, nullptr);
		vkDestroyCommandPool(_logicalDevice, _graphics.commandPool, nullptr);
		vkDestroyCommandPool(_logicalDevice, _compute.commandPool, nullptr);
		vkDestroyDevice(_logicalDevice, nullptr);
		vkDestroySurfaceKHR(_instance, _surface, nullptr);
		vkDestroyInstance(_instance, nullptr);
	}

	void RayTracer::drawFrame() {
		vkWaitForFences(_logicalDevice, 1, &_sync.computeComplete, VK_TRUE, UINT64_MAX);
		vkResetFences(_logicalDevice, 1, &_sync.computeComplete);

		VkSubmitInfo computeSubmitInfo{};
		computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		computeSubmitInfo.commandBufferCount = 1;
		computeSubmitInfo.pCommandBuffers = &_compute.commandBuffer;

		if (vkQueueSubmit(_compute.queue, 1, &computeSubmitInfo, _sync.computeComplete) != VK_SUCCESS) {
			throw std::runtime_error("Failed to submit the compute job");
		}

		uint32_t imageIndex;
		vkAcquireNextImageKHR(_logicalDevice, _swapChain.swapChain, UINT64_MAX, _sync.presentComplete, (VkFence) nullptr, &imageIndex);

		VkPipelineStageFlags waitStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.pWaitDstStageMask = &waitStages;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &_sync.presentComplete;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &_sync.renderComplete;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &_graphics.drawCommandBuffers[imageIndex];

		if (vkQueueSubmit(_graphics.queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
			throw std::runtime_error("Failed to submit the render job");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.pNext = NULL;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &_swapChain.swapChain;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pWaitSemaphores = &_sync.renderComplete;
		presentInfo.waitSemaphoreCount = 1;

		vkQueuePresentKHR(_graphics.queue, &presentInfo);

		if (vkQueueWaitIdle(_graphics.queue) != VK_SUCCESS) {
			throw std::runtime_error("Render job failed");
		}
	}

	void RayTracer::updateSettings(Settings& settings) {
		memcpy(_scene.settingHandle, &settings, sizeof(Settings));
	}

	void RayTracer::createInstance() {
		VkApplicationInfo applicationInfo{};
		applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		applicationInfo.pApplicationName = "Vulkan Ray Tracing";
		applicationInfo.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
		applicationInfo.pEngineName = "No engine";
		applicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		applicationInfo.apiVersion = VK_API_VERSION_1_3;

		uint32_t enabledExtensionCount;
		const char** enabledExtensionNames = glfwGetRequiredInstanceExtensions(&enabledExtensionCount);

		VkInstanceCreateInfo instanceCreateInfo{};
		instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instanceCreateInfo.pApplicationInfo = &applicationInfo;
		instanceCreateInfo.enabledExtensionCount = enabledExtensionCount;
		instanceCreateInfo.ppEnabledExtensionNames = enabledExtensionNames;

#ifndef NDEBUG
		uint32_t availableLayerCount;
		vkEnumerateInstanceLayerProperties(&availableLayerCount, nullptr);

		std::vector<VkLayerProperties> availabeLayers{ availableLayerCount };
		vkEnumerateInstanceLayerProperties(&availableLayerCount, availabeLayers.data());

		bool validationLayerAvailable = true;

		for (const auto& required : REQUIRED_LAYERS) {
			bool isRequiredLayerAvailable = false;

			for (const auto& layer : availabeLayers) {
				if (strcmp(layer.layerName, required) == 0) {
					isRequiredLayerAvailable = true;

					break;
				}
			}

			if (!isRequiredLayerAvailable) {
				validationLayerAvailable = false;

				break;
			}
		}

		if (!validationLayerAvailable) {
			throw std::runtime_error("The required validation layers are not available");
		}

		instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(REQUIRED_LAYERS.size());
		instanceCreateInfo.ppEnabledLayerNames = REQUIRED_LAYERS.data();
#else
		instanceCreateInfo.enabledLayerCount = 0;
		instanceCreateInfo.ppEnabledLayerNames = nullptr;
#endif

		if (vkCreateInstance(&instanceCreateInfo, nullptr, &_instance) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the Vulkan instance");
		}

		_window.createWindowSurface(_instance, &_surface);
	}

	void RayTracer::createDevice() {
		uint32_t physicalDeviceCount;
		vkEnumeratePhysicalDevices(_instance, &physicalDeviceCount, nullptr);

		std::vector<VkPhysicalDevice> physicalDevices{ physicalDeviceCount };
		vkEnumeratePhysicalDevices(_instance, &physicalDeviceCount, physicalDevices.data());

		uint8_t bestDeviceQuality = UINT8_MAX;

		for (const auto& physicalDevice : physicalDevices) {
			uint8_t quality = getPhysicalDeviceQuality(physicalDevice);

			if (quality < bestDeviceQuality) {
				bestDeviceQuality = quality;

				_physicalDevice = physicalDevice;
			}
		}

		if (bestDeviceQuality == UINT8_MAX) {
			throw std::runtime_error("Unable to find a device meeting the requirements");
		}

		uint32_t queueFamilyPropertyCount;
		vkGetPhysicalDeviceQueueFamilyProperties(_physicalDevice, &queueFamilyPropertyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilyProperties{ queueFamilyPropertyCount };
		vkGetPhysicalDeviceQueueFamilyProperties(_physicalDevice, &queueFamilyPropertyCount, queueFamilyProperties.data());

		getGraphicsQueueFamilyIndex(queueFamilyProperties, &_queueFamilyIndices.graphics);
		getComputeQueueFamilyIndex(queueFamilyProperties, &_queueFamilyIndices.compute);
		getTransferQueueFamilyIndex(queueFamilyProperties, &_queueFamilyIndices.transfer);

		std::vector<VkDeviceQueueCreateInfo> deviceQueueCreateInfo;
		std::set<uint32_t> uniqueQueueFamilyIndices = {
			_queueFamilyIndices.graphics,
			_queueFamilyIndices.compute,
			_queueFamilyIndices.transfer
		};

		float queuePriority = 1.0f;

		for (uint32_t queueFamilyIndex : uniqueQueueFamilyIndices) {
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;

			deviceQueueCreateInfo.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures requiredFeatures{};
		VkDeviceCreateInfo deviceCreateInfo{};
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceCreateInfo.pEnabledFeatures = &requiredFeatures;
		deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(REQUIRED_EXTENSION_PROPERTIES.size());
		deviceCreateInfo.ppEnabledExtensionNames = REQUIRED_EXTENSION_PROPERTIES.data();
		deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(deviceQueueCreateInfo.size());
		deviceCreateInfo.pQueueCreateInfos = deviceQueueCreateInfo.data();

#ifndef NDEBUG
		deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(REQUIRED_LAYERS.size());
		deviceCreateInfo.ppEnabledLayerNames = REQUIRED_LAYERS.data();
#else
		deviceCreateInfo.enabledLayerCount = 0;
		deviceCreateInfo.ppEnabledLayerNames = nullptr;
#endif

		if (vkCreateDevice(_physicalDevice, &deviceCreateInfo, nullptr, &_logicalDevice) != VK_SUCCESS) {
			throw std::runtime_error("Unable to create the logical device");
		}
	}

	void RayTracer::createCommandPools() {
		VkCommandPoolCreateInfo graphicsCommandPoolCreateInfo{};
		graphicsCommandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		graphicsCommandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		graphicsCommandPoolCreateInfo.queueFamilyIndex = _queueFamilyIndices.graphics;

		if (vkCreateCommandPool(_logicalDevice, &graphicsCommandPoolCreateInfo, nullptr, &_graphics.commandPool) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the graphics command pool");
		}

		VkCommandPoolCreateInfo computeCommandPoolCreateInfo{};
		computeCommandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		computeCommandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		computeCommandPoolCreateInfo.queueFamilyIndex = _queueFamilyIndices.compute;

		if (vkCreateCommandPool(_logicalDevice, &computeCommandPoolCreateInfo, nullptr, &_compute.commandPool) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the graphics command pool");
		}

		vkGetDeviceQueue(_logicalDevice, _queueFamilyIndices.graphics, 0, &_graphics.queue);
		vkGetDeviceQueue(_logicalDevice, _queueFamilyIndices.compute, 0, &_compute.queue);
	}

	void RayTracer::createSwapChain() {
		auto surfaceFormat = selectSurfaceFormat();
		auto presentMode = selectPresentMode();
		auto surfaceCapabilities = getSurfaceCapabilities();

		_swapChain.imageCount = surfaceCapabilities.minImageCount + 1;

		if (surfaceCapabilities.maxImageCount > 0 && _swapChain.imageCount > surfaceCapabilities.maxImageCount) {
			_swapChain.imageCount = surfaceCapabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR swapChainCreateInfo{};
		swapChainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		swapChainCreateInfo.surface = _surface;
		swapChainCreateInfo.minImageCount = _swapChain.imageCount;
		swapChainCreateInfo.imageFormat = surfaceFormat.format;
		swapChainCreateInfo.imageColorSpace = surfaceFormat.colorSpace;
		swapChainCreateInfo.imageExtent = _swapChain.extent;
		swapChainCreateInfo.imageArrayLayers = 1;
		swapChainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		swapChainCreateInfo.preTransform = surfaceCapabilities.currentTransform;
		swapChainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		swapChainCreateInfo.queueFamilyIndexCount = 0;
		swapChainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		swapChainCreateInfo.presentMode = presentMode;
		swapChainCreateInfo.clipped = VK_TRUE;
		swapChainCreateInfo.oldSwapchain = VK_NULL_HANDLE;

		if (surfaceCapabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
			swapChainCreateInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		}

		if (surfaceCapabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) {
			swapChainCreateInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		}

		if (vkCreateSwapchainKHR(_logicalDevice, &swapChainCreateInfo, nullptr, &_swapChain.swapChain) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the swap chain");
		}

		_swapChain.format = surfaceFormat.format;
		_swapChain.images.resize(_swapChain.imageCount);
		_swapChain.imageViews.resize(_swapChain.imageCount);
		_swapChain.frameBuffers.resize(_swapChain.imageCount);

		vkGetSwapchainImagesKHR(_logicalDevice, _swapChain.swapChain, &_swapChain.imageCount, _swapChain.images.data());

		for (uint32_t i = 0; i < _swapChain.imageCount; i++) {
			VkImageViewCreateInfo colorAttachmentViewCreateInfo = {};
			colorAttachmentViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			colorAttachmentViewCreateInfo.image = _swapChain.images[i];
			colorAttachmentViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			colorAttachmentViewCreateInfo.format = _swapChain.format;
			colorAttachmentViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			colorAttachmentViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			colorAttachmentViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			colorAttachmentViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			colorAttachmentViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			colorAttachmentViewCreateInfo.subresourceRange.baseMipLevel = 0;
			colorAttachmentViewCreateInfo.subresourceRange.levelCount = 1;
			colorAttachmentViewCreateInfo.subresourceRange.baseArrayLayer = 0;
			colorAttachmentViewCreateInfo.subresourceRange.layerCount = 1;
			colorAttachmentViewCreateInfo.flags = 0;

			if (vkCreateImageView(_logicalDevice, &colorAttachmentViewCreateInfo, nullptr, &_swapChain.imageViews[i]) != VK_SUCCESS) {
				throw std::runtime_error("Failed to create the image view");
			}
		}

		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = _swapChain.format;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colorAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(_logicalDevice, &renderPassInfo, nullptr, &_swapChain.renderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}

		for (size_t i = 0; i < _swapChain.imageCount; i++) {
			VkImageView attachments[] = { _swapChain.imageViews[i] };

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = _swapChain.renderPass;
			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments = attachments;
			framebufferInfo.width = _swapChain.extent.width;
			framebufferInfo.height = _swapChain.extent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(_logicalDevice, &framebufferInfo, nullptr, &_swapChain.frameBuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("Failed to create the framebuffer");
			}
		}
	}

	void RayTracer::createTargetTexture() {
		createImageAndView(VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _targetTexture.image, _targetTexture.imageDeviceMemory, _targetTexture.imageView, _swapChain.extent.width, _swapChain.extent.height);
		changeImageLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, _targetTexture.image);
		
		// TODO move?
		VkSamplerCreateInfo samplerCreateInfo{};
		samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
		samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
		samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerCreateInfo.mipLodBias = 0.0f;
		samplerCreateInfo.maxAnisotropy = 1.0f;
		samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
		samplerCreateInfo.minLod = 0.0f;
		samplerCreateInfo.maxLod = 0.0f;
		samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

		if (vkCreateSampler(_logicalDevice, &samplerCreateInfo, nullptr, &_sampler) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the texture sampler");
		}
	}

	void RayTracer::createSkyBox() {
		int texWidth, texHeight, texChannels;
		stbi_uc* layers[6];
		for (size_t index = 0; index < 6; index++) {
			layers[index] = stbi_load(SKY_BOX_TEXTURE_PATHS[index], &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		}
		
		if (!layers[0] || !layers[1] || !layers[2] || !layers[3] || !layers[4] || !layers[5]) {
			throw std::runtime_error("Failed to load the skybox texture image!");
		}

		// RGBA to BGRA conversion
		for (int layer = 0; layer < 6; layer++) {
			for (size_t k = 0; k < texWidth * texHeight; k++) {
				auto r = layers[layer][k * 4 + 2];
				layers[layer][k * 4 + 2] = layers[layer][k * 4];
				layers[layer][k * 4] = r;
			}
		}

		VkDeviceSize imageSize = texWidth * texHeight * 4 * 6;
		VkDeviceSize layerSize = texWidth * texHeight * 4;

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingMemory;
		createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, imageSize, stagingBuffer, stagingMemory);
		createCubeMap(VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _skyBox.image, _skyBox.imageDeviceMemory, _skyBox.imageView, texWidth, texHeight);
		changeImageLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, _skyBox.image, 0, VK_ACCESS_TRANSFER_WRITE_BIT, 6);

		void* dataPointer;
		vkMapMemory(_logicalDevice, stagingMemory, 0, imageSize, 0, &dataPointer);

		for (int layer = 0; layer < 6; layer++) {
			memcpy(static_cast<char*>(dataPointer) + (layer * layerSize), layers[layer], static_cast<size_t>(layerSize));
		}

		vkUnmapMemory(_logicalDevice, stagingMemory);

		for (int layer = 0; layer < 6; layer++) {
			stbi_image_free(layers[layer]);
		}

		VkCommandBuffer copyCommandBuffer;
		createCommandBuffers(_graphics.commandPool, &copyCommandBuffer);

		VkBufferImageCopy bufferImageCopy{};
		bufferImageCopy.bufferOffset = 0;
		bufferImageCopy.bufferRowLength = 0;
		bufferImageCopy.bufferImageHeight = 0;
		bufferImageCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		bufferImageCopy.imageSubresource.mipLevel = 0;
		bufferImageCopy.imageSubresource.baseArrayLayer = 0;
		bufferImageCopy.imageSubresource.layerCount = 6;
		bufferImageCopy.imageOffset = { 0, 0, 0 };
		bufferImageCopy.imageExtent = { (uint32_t)texWidth, (uint32_t)texHeight, 1 };

		vkCmdCopyBufferToImage(copyCommandBuffer, stagingBuffer, _skyBox.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bufferImageCopy);
		submitCommandBuffers(_graphics.commandPool, _graphics.queue, &copyCommandBuffer);
		changeImageLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _skyBox.image, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, 6);
	
		vkFreeMemory(_logicalDevice, stagingMemory, nullptr);
		vkDestroyBuffer(_logicalDevice, stagingBuffer, nullptr);
	}

	void RayTracer::createStorageBuffers() {
		createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, sizeof(Settings), _scene.settingBuffer, _scene.settingMemory);
		vkMapMemory(_logicalDevice, _scene.settingMemory, 0, sizeof(Settings), 0, &_scene.settingHandle);

		std::vector<Sphere> spheres{ 0 };

		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				Sphere sphere{};
				sphere.radius = 2.0f;
				sphere.position = { i * 7, 1.0f, j * 7 };
				sphere.albedo = { 0.2f, 0.2f, 0.2f };
				sphere.specular = { 0.8f, 0.8f, 0.8f };

				glm::vec3 color = {
					static_cast <float> (rand()) / static_cast <float> (RAND_MAX),
					static_cast <float> (rand()) / static_cast <float> (RAND_MAX),
					static_cast <float> (rand()) / static_cast <float> (RAND_MAX)
				};

				if (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) < 0.5f) {
					sphere.albedo = color;
					sphere.specular = { 0.1f, 0.1f, 0.1f };
				} else {
					sphere.albedo = { 0.0f, 0.0f, 0.0f };
					sphere.specular = color;
				}

				spheres.push_back(sphere);
			}
		}


		VkDeviceSize spheresBufferSize = spheres.size() * sizeof(Sphere);
		createStorageBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, spheresBufferSize, _scene.sphereBuffer, _scene.sphereMemory, spheres.data());
	
		glm::vec3 x = { -1, 0, 0 };

		std::vector<Plane> planes = {
			// { { 5.0f, 0.0f, 0.0f }, glm::normalize(x), {0.0f, 0.0f, 0.0f}, {0.9f, 0.9f, 0.9f}},
			// { { -5.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, {0.0f, 0.0f, 0.0f}, {0.9f, 0.9f, 0.9f} },
			// { { 0.0f, 0.0f, -5.0f }, { 0.0f, 0.0f, 1.0f }, {0.0f, 0.0f, 0.0f}, {0.9f, 0.9f, 0.9f} },
			// { { 0.0f, 0.0f, 5.0f }, { 0.0f, 0.0f, -1.0f }, {0.0f, 0.0f, 0.0f}, {0.9f, 0.9f, 0.9f} },
			{ { 0.0f, -1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, {1.0f, 1.0f, 1.0f}, {0.3f, 0.3f, 0.3f} },
		};

		VkDeviceSize planesBufferSize = planes.size() * sizeof(Plane);
		createStorageBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, planesBufferSize, _scene.planeBuffer, _scene.planeMemory, planes.data());
	}

	// TODO move descriptor set creation into their respective pipelines
	// TODO note: the descriptor pool has to be created after the swap chain
	void RayTracer::createDescriptorSets() {
		std::vector<VkDescriptorPoolSize> descriptorPoolSizes = {
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 }
		};

		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
		descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptorPoolCreateInfo.poolSizeCount = static_cast<uint32_t>(descriptorPoolSizes.size());
		descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSizes.data();
		descriptorPoolCreateInfo.maxSets = _swapChain.imageCount;

		if (vkCreateDescriptorPool(_logicalDevice, &descriptorPoolCreateInfo, nullptr, &_descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the descriptor pool!");
		}

		VkDescriptorImageInfo descriptorImageInfo{};
		descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		descriptorImageInfo.imageView = _targetTexture.imageView;
		descriptorImageInfo.sampler = _sampler;

		{
			VkDescriptorSetLayoutBinding graphicsDescriptorSetLayoutBinding{};
			graphicsDescriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			graphicsDescriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
			graphicsDescriptorSetLayoutBinding.binding = 0;
			graphicsDescriptorSetLayoutBinding.descriptorCount = 1;

			VkDescriptorSetLayoutCreateInfo graphicsDescriptorSetLayoutCreateInfo{};
			graphicsDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			graphicsDescriptorSetLayoutCreateInfo.bindingCount = 1;
			graphicsDescriptorSetLayoutCreateInfo.pBindings = &graphicsDescriptorSetLayoutBinding;

			if (vkCreateDescriptorSetLayout(_logicalDevice, &graphicsDescriptorSetLayoutCreateInfo, nullptr, &_graphics.descriptorSetLayout) != VK_SUCCESS) {
				throw std::runtime_error("Failed to create the graphics descriptor set layout");
			}

			VkDescriptorSetAllocateInfo graphicsDescriptorSetAllocateInfo{};
			graphicsDescriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			graphicsDescriptorSetAllocateInfo.descriptorPool = _descriptorPool;
			graphicsDescriptorSetAllocateInfo.descriptorSetCount = 1;
			graphicsDescriptorSetAllocateInfo.pSetLayouts = &_graphics.descriptorSetLayout;

			if (vkAllocateDescriptorSets(_logicalDevice, &graphicsDescriptorSetAllocateInfo, &_graphics.descriptorSet) != VK_SUCCESS) {
				throw std::runtime_error("Failed to allocate the graphics descriptor set");
			}

			VkWriteDescriptorSet graphicsWriteDescriptorSet{};
			graphicsWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			graphicsWriteDescriptorSet.dstSet = _graphics.descriptorSet;
			graphicsWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			graphicsWriteDescriptorSet.dstBinding = 0;
			graphicsWriteDescriptorSet.pImageInfo = &descriptorImageInfo;
			graphicsWriteDescriptorSet.descriptorCount = 1;

			vkUpdateDescriptorSets(_logicalDevice, 1, &graphicsWriteDescriptorSet, 0, nullptr);
		}

		{

			// TODO cleanup
			std::vector<VkDescriptorSetLayoutBinding> computeDescriptorSetLayoutBindings{ 5 };
			VkDescriptorSetLayoutBinding computeSkyBoxDescriptorSetLayoutBinding{};
			computeSkyBoxDescriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			computeSkyBoxDescriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			computeSkyBoxDescriptorSetLayoutBinding.binding = 0;
			computeSkyBoxDescriptorSetLayoutBinding.descriptorCount = 1;
			computeDescriptorSetLayoutBindings[0] = computeSkyBoxDescriptorSetLayoutBinding;

			VkDescriptorSetLayoutBinding computeStorageDescriptorSetLayoutBinding{};
			computeStorageDescriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			computeStorageDescriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			computeStorageDescriptorSetLayoutBinding.binding = 1;
			computeStorageDescriptorSetLayoutBinding.descriptorCount = 1;
			computeDescriptorSetLayoutBindings[1] = computeStorageDescriptorSetLayoutBinding;

			VkDescriptorSetLayoutBinding computeCameraDescriptorSetLayoutBinding{};
			computeCameraDescriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			computeCameraDescriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			computeCameraDescriptorSetLayoutBinding.binding = 2;
			computeCameraDescriptorSetLayoutBinding.descriptorCount = 1;
			computeDescriptorSetLayoutBindings[2] = computeCameraDescriptorSetLayoutBinding;

			VkDescriptorSetLayoutBinding computeSpheresDescriptorSetLayoutBinding{};
			computeSpheresDescriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			computeSpheresDescriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			computeSpheresDescriptorSetLayoutBinding.binding = 3;
			computeSpheresDescriptorSetLayoutBinding.descriptorCount = 1;
			computeDescriptorSetLayoutBindings[3] = computeSpheresDescriptorSetLayoutBinding;

			VkDescriptorSetLayoutBinding computePlanesDescriptorSetLayoutBinding{};
			computePlanesDescriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			computePlanesDescriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			computePlanesDescriptorSetLayoutBinding.binding = 4;
			computePlanesDescriptorSetLayoutBinding.descriptorCount = 1;
			computeDescriptorSetLayoutBindings[4] = computePlanesDescriptorSetLayoutBinding;

			VkDescriptorSetLayoutCreateInfo computeDescriptorSetLayoutCreateInfo{};
			computeDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			computeDescriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(computeDescriptorSetLayoutBindings.size());
			computeDescriptorSetLayoutCreateInfo.pBindings = computeDescriptorSetLayoutBindings.data();

			if (vkCreateDescriptorSetLayout(_logicalDevice, &computeDescriptorSetLayoutCreateInfo, nullptr, &_compute.descriptorSetLayout) != VK_SUCCESS) {
				throw std::runtime_error("Failed to create the compute descriptor set layout");
			}

			VkDescriptorSetAllocateInfo computeDescriptorSetAllocateInfo{};
			computeDescriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			computeDescriptorSetAllocateInfo.descriptorPool = _descriptorPool;
			computeDescriptorSetAllocateInfo.descriptorSetCount = 1;
			computeDescriptorSetAllocateInfo.pSetLayouts = &_compute.descriptorSetLayout;

			if (vkAllocateDescriptorSets(_logicalDevice, &computeDescriptorSetAllocateInfo, &_compute.descriptorSet) != VK_SUCCESS) {
				throw std::runtime_error("Failed to allocate the compute descriptor set");
			}

			VkDescriptorImageInfo skyBoxDescriptorImageInfo{};
			skyBoxDescriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			skyBoxDescriptorImageInfo.imageView = _skyBox.imageView;
			skyBoxDescriptorImageInfo.sampler = _sampler;

			// TODO cleanup
			std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets{ 5 };
			VkWriteDescriptorSet computeSkyBoxWriteDescriptorSet{};
			computeSkyBoxWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			computeSkyBoxWriteDescriptorSet.dstSet = _compute.descriptorSet;
			computeSkyBoxWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			computeSkyBoxWriteDescriptorSet.dstBinding = 0;
			computeSkyBoxWriteDescriptorSet.pImageInfo = &skyBoxDescriptorImageInfo;
			computeSkyBoxWriteDescriptorSet.descriptorCount = 1;
			computeWriteDescriptorSets[0] = computeSkyBoxWriteDescriptorSet;

			VkWriteDescriptorSet computeStorageWriteDescriptorSet{};
			computeStorageWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			computeStorageWriteDescriptorSet.dstSet = _compute.descriptorSet;
			computeStorageWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			computeStorageWriteDescriptorSet.dstBinding = 1;
			computeStorageWriteDescriptorSet.pImageInfo = &descriptorImageInfo;
			computeStorageWriteDescriptorSet.descriptorCount = 1;
			computeWriteDescriptorSets[1] = computeStorageWriteDescriptorSet;

			VkDescriptorBufferInfo cameraDescriptorBufferInfo{};
			cameraDescriptorBufferInfo.buffer = _scene.settingBuffer;
			cameraDescriptorBufferInfo.range = VK_WHOLE_SIZE;
			cameraDescriptorBufferInfo.offset = 0;

			VkWriteDescriptorSet computeCameraWriteDescriptorSet{};
			computeCameraWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			computeCameraWriteDescriptorSet.dstSet = _compute.descriptorSet;
			computeCameraWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			computeCameraWriteDescriptorSet.dstBinding = 2;
			computeCameraWriteDescriptorSet.pBufferInfo = &cameraDescriptorBufferInfo;
			computeCameraWriteDescriptorSet.descriptorCount = 1;
			computeWriteDescriptorSets[2] = computeCameraWriteDescriptorSet;

			VkDescriptorBufferInfo sphereDescriptorBufferInfo{};
			sphereDescriptorBufferInfo.buffer = _scene.sphereBuffer;
			sphereDescriptorBufferInfo.range = VK_WHOLE_SIZE;
			sphereDescriptorBufferInfo.offset = 0;

			VkWriteDescriptorSet computeSpheresWriteDescriptorSet{};
			computeSpheresWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			computeSpheresWriteDescriptorSet.dstSet = _compute.descriptorSet;
			computeSpheresWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			computeSpheresWriteDescriptorSet.dstBinding = 3;
			computeSpheresWriteDescriptorSet.pBufferInfo = &sphereDescriptorBufferInfo;
			computeSpheresWriteDescriptorSet.descriptorCount = 1;
			computeWriteDescriptorSets[3] = computeSpheresWriteDescriptorSet;

			VkDescriptorBufferInfo planeDescriptorBufferInfo{};
			planeDescriptorBufferInfo.buffer = _scene.planeBuffer;
			planeDescriptorBufferInfo.range = VK_WHOLE_SIZE;
			planeDescriptorBufferInfo.offset = 0;

			VkWriteDescriptorSet computePlanesWriteDescriptorSet{};
			computePlanesWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			computePlanesWriteDescriptorSet.dstSet = _compute.descriptorSet;
			computePlanesWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			computePlanesWriteDescriptorSet.dstBinding = 4;
			computePlanesWriteDescriptorSet.pBufferInfo = &planeDescriptorBufferInfo;
			computePlanesWriteDescriptorSet.descriptorCount = 1;
			computeWriteDescriptorSets[4] = computePlanesWriteDescriptorSet;

			vkUpdateDescriptorSets(_logicalDevice, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, nullptr);
		}
	}

	void RayTracer::createGraphicsPipeline() {
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &_graphics.descriptorSetLayout;
		pipelineLayoutInfo.pushConstantRangeCount = 0;
		pipelineLayoutInfo.pPushConstantRanges = nullptr;

		if (vkCreatePipelineLayout(_logicalDevice, &pipelineLayoutInfo, nullptr, &_graphics.pipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the pipeline layout");
		}

		VkShaderModule shaderVertex{};
		loadShaderModule(SHADER_VERTEX_PATH, shaderVertex);

		VkPipelineShaderStageCreateInfo vertexShaderStageInfo{};
		vertexShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertexShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertexShaderStageInfo.module = shaderVertex;
		vertexShaderStageInfo.pName = "main";

		VkShaderModule shaderFragment;
		loadShaderModule(SHADER_FRAGMENT_PATH, shaderFragment);

		VkPipelineShaderStageCreateInfo fragmentShaderStageInfo{};
		fragmentShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragmentShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragmentShaderStageInfo.module = shaderFragment;
		fragmentShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertexShaderStageInfo, fragmentShaderStageInfo };

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 0;
		vertexInputInfo.pVertexBindingDescriptions = nullptr;
		vertexInputInfo.vertexAttributeDescriptionCount = 0;
		vertexInputInfo.pVertexAttributeDescriptions = nullptr;

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)_swapChain.extent.width;
		viewport.height = (float)_swapChain.extent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = _swapChain.extent;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		VkGraphicsPipelineCreateInfo graphicsPipelineInfo{};
		graphicsPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		graphicsPipelineInfo.stageCount = 2;
		graphicsPipelineInfo.pStages = shaderStages;
		graphicsPipelineInfo.pVertexInputState = &vertexInputInfo;
		graphicsPipelineInfo.pInputAssemblyState = &inputAssembly;
		graphicsPipelineInfo.pViewportState = &viewportState;
		graphicsPipelineInfo.pRasterizationState = &rasterizer;
		graphicsPipelineInfo.pMultisampleState = &multisampling;
		graphicsPipelineInfo.pDepthStencilState = nullptr;
		graphicsPipelineInfo.pColorBlendState = &colorBlending;
		graphicsPipelineInfo.pDynamicState = nullptr;
		graphicsPipelineInfo.layout = _graphics.pipelineLayout;
		graphicsPipelineInfo.renderPass = _swapChain.renderPass;
		graphicsPipelineInfo.subpass = 0;
		graphicsPipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		graphicsPipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(_logicalDevice, VK_NULL_HANDLE, 1, &graphicsPipelineInfo, nullptr, &_graphics.pipeline) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the graphics pipeline");
		}

		vkDestroyShaderModule(_logicalDevice, shaderVertex, nullptr);
		vkDestroyShaderModule(_logicalDevice, shaderFragment, nullptr);
	}

	void RayTracer::createComputePipeline() {
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &_compute.descriptorSetLayout;
		pipelineLayoutInfo.pushConstantRangeCount = 0;
		pipelineLayoutInfo.pPushConstantRanges = nullptr;

		if (vkCreatePipelineLayout(_logicalDevice, &pipelineLayoutInfo, nullptr, &_compute.pipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the pipeline layout");
		}

		VkShaderModule shaderCompute{};
		loadShaderModule(SHADER_COMPUTE_PATH, shaderCompute);

		VkPipelineShaderStageCreateInfo vertexComputeStageInfo{};
		vertexComputeStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertexComputeStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		vertexComputeStageInfo.module = shaderCompute;
		vertexComputeStageInfo.pName = "main";

		VkComputePipelineCreateInfo computePipelineCreateInfo{};
		computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		computePipelineCreateInfo.layout = _compute.pipelineLayout;
		computePipelineCreateInfo.flags = 0;
		computePipelineCreateInfo.stage = vertexComputeStageInfo;

		if (vkCreateComputePipelines(_logicalDevice, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &_compute.pipeline) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the compute pipeline");
		}

		vkDestroyShaderModule(_logicalDevice, shaderCompute, nullptr);
	}

	void RayTracer::createDrawCommandBuffers() {
		_graphics.drawCommandBuffers.resize(_swapChain.imageCount);
		createCommandBuffers(_graphics.commandPool, _graphics.drawCommandBuffers.data(), _swapChain.imageCount);

		VkClearValue clearValues[2];
		clearValues[0].color = { 0.1f, 0.1f, 0.1f, 1.0f };
		clearValues[1].depthStencil = { 1.0f, 0 };

		for (uint32_t i = 0; i < _swapChain.imageCount; i++) {
			VkImageMemoryBarrier imageMemoryBarrier = {};
			imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.image = _targetTexture.image;
			imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

			if (_queueFamilyIndices.graphics != _queueFamilyIndices.compute) {
				imageMemoryBarrier.srcAccessMask = 0;
				imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
				imageMemoryBarrier.srcQueueFamilyIndex = _queueFamilyIndices.compute;
				imageMemoryBarrier.dstQueueFamilyIndex = _queueFamilyIndices.graphics;

				vkCmdPipelineBarrier(_graphics.drawCommandBuffers[i], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
			} else {
				imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
				imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
				imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

				vkCmdPipelineBarrier(_graphics.drawCommandBuffers[i], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
			}

			VkRenderPassBeginInfo renderPassBeginInfo{};
			renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassBeginInfo.renderPass = _swapChain.renderPass;
			renderPassBeginInfo.renderArea.offset = { 0, 0 };
			renderPassBeginInfo.renderArea.extent = _swapChain.extent;
			renderPassBeginInfo.clearValueCount = 2;
			renderPassBeginInfo.pClearValues = clearValues;
			renderPassBeginInfo.framebuffer = _swapChain.frameBuffers[i];

			vkCmdBeginRenderPass(_graphics.drawCommandBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdBindPipeline(_graphics.drawCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, _graphics.pipeline);
			vkCmdBindDescriptorSets(_graphics.drawCommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, _graphics.pipelineLayout, 0, 1, &_graphics.descriptorSet, 0, nullptr);
			vkCmdDraw(_graphics.drawCommandBuffers[i], 3, 1, 0, 0);
			vkCmdEndRenderPass(_graphics.drawCommandBuffers[i]);

			if (_queueFamilyIndices.graphics != _queueFamilyIndices.compute) {
				imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
				imageMemoryBarrier.dstAccessMask = 0;
				imageMemoryBarrier.srcQueueFamilyIndex = _queueFamilyIndices.graphics;
				imageMemoryBarrier.dstQueueFamilyIndex = _queueFamilyIndices.compute;

				vkCmdPipelineBarrier(_graphics.drawCommandBuffers[i], VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
			}

			if (vkEndCommandBuffer(_graphics.drawCommandBuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("Failed to record the command buffer!");
			}
		}
	}

	void RayTracer::createComputeCommandBuffer() {
		createCommandBuffers(_compute.commandPool, &_compute.commandBuffer);

		if (_queueFamilyIndices.graphics != _queueFamilyIndices.compute) {
			VkImageMemoryBarrier imageMemoryBarrier = {};
			imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.image = _targetTexture.image;
			imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
			imageMemoryBarrier.srcAccessMask = 0;
			imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			imageMemoryBarrier.srcQueueFamilyIndex = _queueFamilyIndices.graphics;
			imageMemoryBarrier.dstQueueFamilyIndex = _queueFamilyIndices.compute;

			vkCmdPipelineBarrier(_compute.commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
		}

		vkCmdBindPipeline(_compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _compute.pipeline);
		vkCmdBindDescriptorSets(_compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _compute.pipelineLayout, 0, 1, &_compute.descriptorSet, 0, 0);
		vkCmdDispatch(_compute.commandBuffer, _swapChain.extent.width / 16, _swapChain.extent.height / 16, 1);

		if (_queueFamilyIndices.graphics != _queueFamilyIndices.compute) {
			VkImageMemoryBarrier imageMemoryBarrier = {};
			imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.image = _targetTexture.image;
			imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
			imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			imageMemoryBarrier.dstAccessMask = 0;
			imageMemoryBarrier.srcQueueFamilyIndex = _queueFamilyIndices.compute;
			imageMemoryBarrier.dstQueueFamilyIndex = _queueFamilyIndices.graphics;

			vkCmdPipelineBarrier(_compute.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
		}

		if (vkEndCommandBuffer(_compute.commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("Failed to end the recording of the compute command buffer");
		}
	}

	void RayTracer::createSemaphoresAndFences() {
		VkSemaphoreCreateInfo semaphoreCreateInfo{};
		semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		if (vkCreateSemaphore(_logicalDevice, &semaphoreCreateInfo, nullptr, &_sync.presentComplete) != VK_SUCCESS ||
			vkCreateSemaphore(_logicalDevice, &semaphoreCreateInfo, nullptr, &_sync.renderComplete) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the semaphores");
		}

		VkFenceCreateInfo fenceCreateInfo{};
		fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		if (vkCreateFence(_logicalDevice, &fenceCreateInfo, nullptr, &_sync.computeComplete) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the compute shader fence");
		}

		if (_queueFamilyIndices.graphics != _queueFamilyIndices.compute) {


			VkCommandBuffer commandBuffer;
			createCommandBuffers(_graphics.commandPool, &commandBuffer);

			VkImageMemoryBarrier imageMemoryBarrier = {};
			imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.image = _targetTexture.image;
			imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
			imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			imageMemoryBarrier.dstAccessMask = 0;
			imageMemoryBarrier.srcQueueFamilyIndex = _queueFamilyIndices.graphics;
			imageMemoryBarrier.dstQueueFamilyIndex = _queueFamilyIndices.compute;
			
			vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
			submitCommandBuffers(_graphics.commandPool, _graphics.queue, &commandBuffer);
		}
	}

	uint8_t RayTracer::getPhysicalDeviceQuality(VkPhysicalDevice physicalDevice) {
		uint32_t extensionPropertyCount;
		vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionPropertyCount, nullptr);

		std::vector<VkExtensionProperties> extensionProperties{ extensionPropertyCount };
		vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionPropertyCount, extensionProperties.data());

		bool hasRequiredExtensions = true;

		for (const auto& requiredExtensionProperty : REQUIRED_EXTENSION_PROPERTIES) {
			bool hasExtension = false;

			for (const auto& extensionProperty : extensionProperties) {
				if (strcmp(extensionProperty.extensionName, requiredExtensionProperty) == 0) {
					hasExtension = true;

					break;
				}
			}

			if (!hasExtension) {
				hasRequiredExtensions = false;

				break;
			}
		}

		if (!hasRequiredExtensions) {
			return UINT8_MAX;
		}

		uint32_t surfaceFormatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, _surface, &surfaceFormatCount, nullptr);

		if (surfaceFormatCount == 0) {
			return UINT8_MAX;
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, _surface, &presentModeCount, nullptr);

		if (presentModeCount == 0) {
			return UINT8_MAX;
		}

		uint32_t queueFamilyPropertyCount;
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilyProperties{ queueFamilyPropertyCount };
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, queueFamilyProperties.data());

		uint32_t graphicsQueueFamilyIndex;
		if (!getGraphicsQueueFamilyIndex(queueFamilyProperties, &graphicsQueueFamilyIndex)) {
			return UINT8_MAX;
		}

		uint32_t computeQueueFamilyIndex;
		if (!getComputeQueueFamilyIndex(queueFamilyProperties, &computeQueueFamilyIndex)) {
			return UINT8_MAX;
		}

		uint32_t transferQueueFamilyIndex;
		if (!getTransferQueueFamilyIndex(queueFamilyProperties, &transferQueueFamilyIndex)) {
			return UINT8_MAX;
		}

		uint8_t quality{ 0 };

		if (graphicsQueueFamilyIndex == computeQueueFamilyIndex) {
			quality += 3;
		}

		if (graphicsQueueFamilyIndex == transferQueueFamilyIndex) {
			quality += 1;
		}

		if (computeQueueFamilyIndex == transferQueueFamilyIndex) {
			quality += 1;
		}

		VkPhysicalDeviceProperties physicalDeviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

		switch (physicalDeviceProperties.deviceType) {
		case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return quality;
		case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return quality + 0x10;
		case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return quality + 0x20;
		case VK_PHYSICAL_DEVICE_TYPE_CPU: return quality + 0x30;
		default: return quality + 0x40;
		}
	}

	bool RayTracer::getGraphicsQueueFamilyIndex(std::vector<VkQueueFamilyProperties>& queueFamilyProperties, uint32_t* queueFamilyIndex) {
		for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++) {
			if (queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				*queueFamilyIndex = i;

				return true;
			}
		}

		return false;
	}

	bool RayTracer::getComputeQueueFamilyIndex(std::vector<VkQueueFamilyProperties>& queueFamilyProperties, uint32_t* queueFamilyIndex) {
		bool hasFoundAny{ false };

		for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++) {
			if (queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT && (queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
				*queueFamilyIndex = i;

				return true;
			} else if (queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
				*queueFamilyIndex = i;

				hasFoundAny = true;
			}
		}

		return hasFoundAny;
	}

	bool RayTracer::getTransferQueueFamilyIndex(std::vector<VkQueueFamilyProperties>& queueFamilyProperties, uint32_t* queueFamilyIndex) {
		bool hasFoundAny{ false };

		for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++) {
			if (queueFamilyProperties[i].queueFlags & VK_QUEUE_TRANSFER_BIT && (queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0 && (queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
				*queueFamilyIndex = i;

				return true;
			} else if (queueFamilyProperties[i].queueFlags & VK_QUEUE_TRANSFER_BIT) {
				*queueFamilyIndex = i;

				hasFoundAny = true;
			}
		}

		return hasFoundAny;
	}

	VkSurfaceFormatKHR RayTracer::selectSurfaceFormat() {
		uint32_t surfaceFormatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(_physicalDevice, _surface, &surfaceFormatCount, nullptr);

		std::vector<VkSurfaceFormatKHR> surfaceFormats{ surfaceFormatCount };
		vkGetPhysicalDeviceSurfaceFormatsKHR(_physicalDevice, _surface, &surfaceFormatCount, surfaceFormats.data());

		for (auto const& availableFormat : surfaceFormats) {
			VkFormatProperties formatProperties;
			vkGetPhysicalDeviceFormatProperties(_physicalDevice, availableFormat.format, &formatProperties);

			if (formatProperties.optimalTilingFeatures& VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}

		return surfaceFormats[0];
	}

	VkPresentModeKHR RayTracer::selectPresentMode() {
		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(_physicalDevice, _surface, &presentModeCount, nullptr);

		std::vector<VkPresentModeKHR> presentModes{ presentModeCount };
		vkGetPhysicalDeviceSurfacePresentModesKHR(_physicalDevice, _surface, &presentModeCount, presentModes.data());

		for (auto const& availablePresentMode : presentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return VK_PRESENT_MODE_MAILBOX_KHR;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkSurfaceCapabilitiesKHR RayTracer::getSurfaceCapabilities() {
		VkSurfaceCapabilitiesKHR surfaceCapabilities;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(_physicalDevice, _surface, &surfaceCapabilities);

		_swapChain.extent = surfaceCapabilities.currentExtent;

		if (_swapChain.extent.width == std::numeric_limits<uint32_t>::max()) {
			int width, height;

			_window.getFrameBufferSize(&width, &height);

			_swapChain.extent.width = static_cast<uint32_t>(width);
			_swapChain.extent.height = static_cast<uint32_t>(height);

			if (_swapChain.extent.width < surfaceCapabilities.minImageExtent.width) {
				_swapChain.extent.width = surfaceCapabilities.minImageExtent.width;
			} else if (_swapChain.extent.width > surfaceCapabilities.maxImageExtent.width) {
				_swapChain.extent.width = surfaceCapabilities.maxImageExtent.width;
			}

			if (_swapChain.extent.height < surfaceCapabilities.minImageExtent.height) {
				_swapChain.extent.height = surfaceCapabilities.minImageExtent.height;
			} else if (_swapChain.extent.height > surfaceCapabilities.maxImageExtent.height) {
				_swapChain.extent.height = surfaceCapabilities.maxImageExtent.height;
			}
		}

		return surfaceCapabilities;
	}

	uint32_t RayTracer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
		vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &physicalDeviceMemoryProperties);

		for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("Could not find a matching memory type");
	}

	void RayTracer::createCommandBuffers(VkCommandPool commandPool, VkCommandBuffer* commandBuffers, uint32_t commandBufferCount) {
		VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
		commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		commandBufferAllocateInfo.commandPool = commandPool;
		commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		commandBufferAllocateInfo.commandBufferCount = commandBufferCount;

		if (vkAllocateCommandBuffers(_logicalDevice, &commandBufferAllocateInfo, commandBuffers) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate the layout command buffer");
		}

		VkCommandBufferBeginInfo commandBufferBeginInfo{};
		commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		for (uint32_t index = 0; index < commandBufferCount; index++) {
			if (vkBeginCommandBuffer(commandBuffers[index], &commandBufferBeginInfo) != VK_SUCCESS) {
				throw std::runtime_error("Failed to record the layout command buffer");
			}
		}
	}

	void RayTracer::submitCommandBuffers(VkCommandPool commandPool, VkQueue queue, VkCommandBuffer* commandBuffers, uint32_t commandBufferCount) {
		for (uint32_t index = 0; index < commandBufferCount; index++) {
			if (vkEndCommandBuffer(commandBuffers[index]) != VK_SUCCESS) {
				throw std::runtime_error("Failed to record the layout command buffer");
			}
		}

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = commandBuffers;

		VkFence fence;
		VkFenceCreateInfo fenceCreateInfo{};
		fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCreateInfo.flags = 0;

		if (vkCreateFence(_logicalDevice, &fenceCreateInfo, nullptr, &fence) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the transition fence");
		}

		if (vkQueueSubmit(queue, 1, &submitInfo, fence) != VK_SUCCESS) {
			throw std::runtime_error("Failed to submit the transition fence");
		}

		vkWaitForFences(_logicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);
		vkDestroyFence(_logicalDevice, fence, nullptr);
		vkFreeCommandBuffers(_logicalDevice, commandPool, commandBufferCount, commandBuffers);
	}

	void RayTracer::createBuffer(VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& memory) {
		VkBufferCreateInfo bufferCreateInfo{};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.size = size;
		bufferCreateInfo.usage = usage;
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(_logicalDevice, &bufferCreateInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the buffer");
		}

		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(_logicalDevice, buffer, &memoryRequirements);

		VkMemoryAllocateInfo memoryAllocateInfo{};
		memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memoryAllocateInfo.allocationSize = memoryRequirements.size;
		memoryAllocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(_logicalDevice, &memoryAllocateInfo, nullptr, &memory) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate the buffer memory");
		}

		if (vkBindBufferMemory(_logicalDevice, buffer, memory, 0) != VK_SUCCESS) {
			throw std::runtime_error("Failed to bind the buffer memory");
		}
	}

	void RayTracer::createImageAndView(VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& memory, VkImageView& view, uint32_t width, uint32_t height) {
		VkDeviceSize size = width * height * 4;

		VkImageCreateInfo imageCreateInfo{};
		imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
		imageCreateInfo.format = _swapChain.format;
		imageCreateInfo.extent = { width, height, 1 };
		imageCreateInfo.mipLevels = 1;
		imageCreateInfo.arrayLayers = 1;
		imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageCreateInfo.usage = usage;
		imageCreateInfo.flags = 0;

		if (vkCreateImage(_logicalDevice, &imageCreateInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the image");
		}

		VkMemoryRequirements memoryRequirements;
		vkGetImageMemoryRequirements(_logicalDevice, image, &memoryRequirements);

		VkMemoryAllocateInfo memoryAllocateInfo{};
		memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memoryAllocateInfo.allocationSize = memoryRequirements.size;
		memoryAllocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(_logicalDevice, &memoryAllocateInfo, nullptr, &memory) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate the image memory");
		}

		if (vkBindImageMemory(_logicalDevice, image, memory, 0) != VK_SUCCESS) {
			throw std::runtime_error("Failed to bind the image memory");
		}

		VkImageViewCreateInfo imageViewCreateInfo{};
		imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCreateInfo.format = _swapChain.format;
		imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCreateInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		imageViewCreateInfo.image = image;

		if (vkCreateImageView(_logicalDevice, &imageViewCreateInfo, nullptr, &view) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the image view");
		}
	}

	void RayTracer::createCubeMap(VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& memory, VkImageView& view, uint32_t width, uint32_t height) {
		VkImageCreateInfo imageCreateInfo{};
		imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
		imageCreateInfo.format = _swapChain.format;
		imageCreateInfo.extent = { width, height, 1 };
		imageCreateInfo.mipLevels = 1;
		imageCreateInfo.arrayLayers = 6;
		imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageCreateInfo.usage = usage;
		imageCreateInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

		if (vkCreateImage(_logicalDevice, &imageCreateInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create sky box image!");
		}

		VkMemoryRequirements memoryRequirements{};
		vkGetImageMemoryRequirements(_logicalDevice, image, &memoryRequirements);

		VkMemoryAllocateInfo memoryAllocateInfo{};
		memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memoryAllocateInfo.allocationSize = memoryRequirements.size;
		memoryAllocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(_logicalDevice, &memoryAllocateInfo, nullptr, &memory) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate sky box image memory!");
		}

		if (vkBindImageMemory(_logicalDevice, image, memory, 0) != VK_SUCCESS) {
			throw std::runtime_error("Failed to bind the sky box image memory");
		}

		VkImageViewCreateInfo imageViewCreateInfo{};
		imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
		imageViewCreateInfo.format = _swapChain.format;
		imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCreateInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		imageViewCreateInfo.subresourceRange.layerCount = 6;
		imageViewCreateInfo.image = image;

		if (vkCreateImageView(_logicalDevice, &imageViewCreateInfo, nullptr, &view) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the image view");
		}
	}

	void RayTracer::changeImageLayout(VkImageLayout oldLayout, VkImageLayout newLayout, VkImage image, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, uint32_t layerCount) {
		VkCommandBuffer layoutCommandBuffer;
		createCommandBuffers(_graphics.commandPool, &layoutCommandBuffer);

		VkImageMemoryBarrier imageMemoryBarrier{};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageMemoryBarrier.oldLayout = oldLayout;
		imageMemoryBarrier.newLayout = newLayout;
		imageMemoryBarrier.image = image;
		imageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageMemoryBarrier.subresourceRange.baseMipLevel = 0;
		imageMemoryBarrier.subresourceRange.levelCount = 1;
		imageMemoryBarrier.subresourceRange.layerCount = layerCount;
		imageMemoryBarrier.srcAccessMask = srcAccessMask;
		imageMemoryBarrier.dstAccessMask = dstAccessMask;

		vkCmdPipelineBarrier(layoutCommandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
		submitCommandBuffers(_graphics.commandPool, _graphics.queue, &layoutCommandBuffer);
	}

	void RayTracer::createStorageBuffer(VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& bufferMemory, void* data) {
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, size, stagingBuffer, stagingBufferMemory);
		createBuffer(usage, properties, size, buffer, bufferMemory);

		void* dataPointer;
		vkMapMemory(_logicalDevice, stagingBufferMemory, 0, size, 0, &dataPointer);
		memcpy(dataPointer, data, (size_t)size);
		vkUnmapMemory(_logicalDevice, stagingBufferMemory);

		VkCommandBuffer copyCommandBuffer;
		createCommandBuffers(_graphics.commandPool, &copyCommandBuffer);

		VkBufferCopy bufferCopy{};
		bufferCopy.size = size;

		vkCmdCopyBuffer(copyCommandBuffer, stagingBuffer, buffer, 1, &bufferCopy);
		submitCommandBuffers(_graphics.commandPool, _graphics.queue, &copyCommandBuffer);

		vkFreeMemory(_logicalDevice, stagingBufferMemory, nullptr);
		vkDestroyBuffer(_logicalDevice, stagingBuffer, nullptr);
	}

	void RayTracer::loadShaderModule(const char* path, VkShaderModule& shaderModule) {
		std::ifstream file(path, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error("Failed to read the shader file");
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> shaderCode(fileSize);

		file.seekg(0);
		file.read(shaderCode.data(), fileSize);
		file.close();

		VkShaderModuleCreateInfo shaderModuleCreateInfo{};
		shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		shaderModuleCreateInfo.codeSize = shaderCode.size();
		shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());

		if (vkCreateShaderModule(_logicalDevice, &shaderModuleCreateInfo, nullptr, &shaderModule) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create the shader module");
		}
	}
}