#version 450

layout (binding = 0) uniform sampler2D samplerColor;

layout (location = 0) in vec2 texturePosition;

layout (location = 0) out vec4 outFragColor;

void main() {
    outFragColor = texture(samplerColor, vec2(texturePosition.s, 1.0 - texturePosition.t));
	
	if(texturePosition.s > 1) { 
		outFragColor.y = 1.0;
	}
}