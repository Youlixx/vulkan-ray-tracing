#include "vrt_camera.hpp"

#include <glm/gtx/transform.hpp>

namespace vrt {
    Camera::Camera(float fov, float aspect) : _position{ 0.0f }, _rotation{ 0.0f, 0.0f, 0.0f } {
		const float tanHalfFOV = tan(glm::radians(fov) / 2.0f);
		float far = 10.0f;
		float near = 0.1f;

		_projection = glm::mat4{ 0.0f };
		_projection[0][0] = 1.0f / (aspect * tanHalfFOV);
		_projection[1][1] = 1.0f / tanHalfFOV;
		_projection[2][2] = far / (far - near);
		_projection[2][3] = 1.0f;
		_projection[3][2] = -(far * near) / (far - near);
		_projection = glm::inverse(_projection);

		_position = { 0.0f, 0.0f, 0.0f };
	}

    Camera::~Camera() { }

    void Camera::move(GLFWwindow* window, float dt) {
		glm::vec3 rotation{ 0.0f };

		if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
			rotation.y += 1.0f;
		}

		if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
			rotation.y -= 1.0f;
		}

		if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
			rotation.x += 1.0f;
		}

		if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
			rotation.x -= 1.0f;
		}

		if (glm::dot(rotation, rotation) > std::numeric_limits<float>::epsilon()) {
			_rotation += LOOK_SPEED * glm::normalize(rotation) * dt;
		}

		_rotation.x = glm::clamp(_rotation.x, - glm::pi<float>() / 2.0f, glm::pi<float>() / 2.0f);
		_rotation.y = glm::mod(_rotation.y, glm::two_pi<float>());

		const glm::vec3 forwardDirection{ sin(_rotation.y), 0.0f, cos(_rotation.y) };
		const glm::vec3 rightDirection{ forwardDirection.z, 0.0f, -forwardDirection.x };
		const glm::vec3 upDirection{ 0.0f, 1.0f, 0.0f };

		glm::vec3 moveDir{ 0.0f };

		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
			moveDir += forwardDirection;
		}

		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
			moveDir -= forwardDirection;
		}

		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
			moveDir += rightDirection;
		}

		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
			moveDir -= rightDirection;
		}

		if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
			moveDir += upDirection;
		}

		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
			moveDir -= upDirection;
		}

		if (glm::dot(moveDir, moveDir) > std::numeric_limits<float>::epsilon()) {
			_position += MOVE_SPEED * glm::normalize(moveDir) * dt;
		}
    }

    void Camera::setPerspective(float fov, float aspect) {
        _projection = glm::inverse(glm::perspective(glm::radians(fov), aspect, 0.1f, 10.0f));
    }

	const glm::mat4 Camera::getWorldTransform() const {
        const float c3 = glm::cos(_rotation.z);
        const float s3 = glm::sin(_rotation.z);
        const float c2 = glm::cos(_rotation.x);
        const float s2 = glm::sin(_rotation.x);
        const float c1 = glm::cos(_rotation.y);
        const float s1 = glm::sin(_rotation.y);

        return glm::mat4{
            { (c1 * c3 + s1 * s2 * s3), (c2 * s3), (c1 * s2 * s3 - c3 * s1), 0.0f },
            { (c3 * s1 * s2 - c1 * s3), (c2 * c3), (c1 * c3 * s2 + s1 * s3), 0.0f },
            { (c2 * s1), (-s2), (c1 * c2), 0.0f },
            { _position.x, _position.y, _position.z, 1.0f }
        };
	}

	const glm::mat4& Camera::getProjectionMatrix() const {
		return _projection;
	}
}