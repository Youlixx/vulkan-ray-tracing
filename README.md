# vulkan-ray-tracing

A few updates:
- Fixed skybox texture using the swapchain format instead of RGBA.
- Added cmake file to build the project

![Ray traced spheres](ray_tracing.jpg)

The ray tracing code is based on this [blogpost](http://blog.three-eyed-games.com/2018/05/03/gpu-ray-tracing-in-unity-part-1/).
My first attempt at using Vulkan, the ray tracing algorithm is still very basic.
 - The ray tracing algorithm is executed for each pixel in a compute shader
 - The compute shader writes the color result in a texture
 - The graphics pipeline merely renders the texture on the screen

## How to build
To compile this project, you will need the following libraries
 - Vulkan
 - GLFW
 - [stb (for image loading)](https://github.com/nothings/stb)

Run the following command to build the project
```
mkdir -p build
cd build
cmake ..
make
./vulkan_ray_tracer
```
