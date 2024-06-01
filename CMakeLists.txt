cmake_minimum_required(VERSION 3.5.0)
project(vulkan_ray_tracer CXX)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data/shaders)
file(GLOB SHADERS 
    ${SHADER_DIR}/*.vert 
    ${SHADER_DIR}/*.frag 
    ${SHADER_DIR}/*.comp
)

foreach(SHADER IN LISTS SHADERS)
    get_filename_component(FILENAME ${SHADER} NAME)
    add_custom_command(OUTPUT ${SHADER_DIR}/shaders/${FILENAME}.spv
        COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/shaders/ &&
        ${Vulkan_GLSLC_EXECUTABLE} ${SHADER} 
        -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/${FILENAME}.spv
        DEPENDS ${SHADER}
        COMMENT "Compiling ${FILENAME}"
    )
    list(APPEND SPV_SHADERS ${SHADER_DIR}/shaders/${FILENAME}.spv)
endForeach()

add_custom_target(shaders ALL DEPENDS ${SPV_SHADERS})

include_directories(src/)
include_directories(third_party/stb/)

add_executable(vulkan_ray_tracer 
    src/main.cpp
    src/vrt_camera.cpp
    src/vrt_ray_tracer.cpp
    src/vrt_window.cpp
)

target_link_libraries(vulkan_ray_tracer Vulkan::Vulkan glfw)
