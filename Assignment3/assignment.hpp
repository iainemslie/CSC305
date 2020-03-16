#pragma once

#include "paths.hpp"

#include <exception>
#include <iostream>
#include <string>

#include <atlas/glx/Buffer.hpp>
#include <atlas/glx/Context.hpp>
#include <atlas/glx/ErrorCallback.hpp>
#include <atlas/glx/GLSL.hpp>
#include <atlas/utils/Cameras.hpp>
#include <atlas/utils/LoadObjFile.hpp>

#include <fmt/printf.h>
#include <magic_enum.hpp>

#include <algorithm>

using namespace atlas;

static constexpr float nearVal{1.0f};
static constexpr float farVal{10000000000.0f};

static const std::vector<std::string> IncludeDir{ShaderPath};

struct OpenGLError : std::runtime_error
{
    OpenGLError(const std::string& what_arg) : std::runtime_error(what_arg){};
    OpenGLError(const char* what_arg) : std::runtime_error(what_arg){};
};


// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
enum Camera_Movement {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT
};

// Default camera values
const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 5.0f;
const float SENSITIVITY = 0.1f;
const float ZOOM = 45.0f;

// An abstract camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices ofr use in OpenGl
class Camera
{
public:
	// Camera Attributes
	glm::vec3 mPosition;
	glm::vec3 mFront;
	glm::vec3 mUp;
	glm::vec3 mRight;
	glm::vec3 mWorldUp;
	// Euler Angles
	float mYaw;
	float mPitch;
	// Camera options
	float mMovementSpeed;
	float mMouseSensitivity;

	// Constructor with vectors
	Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH) : mFront(glm::vec3(0.0f, 0.0f, -1.0f)), mMovementSpeed(SPEED), mMouseSensitivity(SENSITIVITY)
	{
		mPosition = position;
		mWorldUp = up;
		mYaw = yaw;
		mPitch = pitch;
		updateCameraVectors();
	}

	// Returns the view matrix calculated using Euler Angles and the LookAt Matrix
	glm::mat4 GetViewMatrix()
	{
		return glm::lookAt(mPosition, mPosition + mFront, mUp);
	}

	// Processes input received from any keyboard-like input system. Accepts input parameters in the form of camera defined ENUM (to abstract it from windowing systems) 
	void ProcessKeyboard(Camera_Movement direction, float deltaTime)
	{
		float velocity = mMovementSpeed * deltaTime;
		if (direction == FORWARD)
			mPosition += mFront * velocity;
		if (direction == BACKWARD)
			mPosition -= mFront * velocity;
		if (direction == LEFT)
			mPosition -= mRight * velocity;
		if (direction == RIGHT)
			mPosition += mRight * velocity;
	}

	// Processes input received from a mouse input system. Expects the offset value in both the x and y direction
	void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true)
	{
		xoffset *= mMouseSensitivity;
		yoffset *= mMouseSensitivity;

		mYaw += xoffset;
		mPitch += yoffset;

		// Make sure that when pitch is out of bounds, screen doesn't get flipped
		if (constrainPitch)
		{
			if (mPitch > 89.0f)
				mPitch = 89.0f;
			if (mPitch < -89.0f)
				mPitch = -89.0f;
		}

		// Update Front, Right, and Up Vectors using the updated Euler angles
		updateCameraVectors();
	}

private:
	// Calculates the front vector from the Camera's (updated) Euler Angles
	void updateCameraVectors()
	{
		// Calculate the new Front vector
		glm::vec3 front;
		front.x = cos(glm::radians(mYaw)) * cos(glm::radians(mPitch));
		front.y = sin(glm::radians(mPitch));
		front.z = sin(glm::radians(mYaw)) * cos(glm::radians(mPitch));
		mFront = glm::normalize(front);
		// Also re-calculate the Right and Up vector
		mRight = glm::normalize(glm::cross(mFront, mWorldUp));	// Normalize the vectors
		mUp = glm::normalize(glm::cross(mRight, mFront));
	}
};

class SceneObject
{
public:
	SceneObject::SceneObject()
	{}

	virtual void loadShaders(std::string vertexShader, std::string fragShader) = 0;

	virtual void loadDataToGPU(std::vector<float> const& vertices, std::vector<GLuint> const& indices) = 0;

	virtual void reloadShaders() = 0;

	virtual void loadNewShaders() = 0;

	virtual void render(bool paused, int width, int height, Camera *camera) = 0;

	virtual void freeGPUData() = 0;

	virtual void setPosition(glm::vec3 position) = 0;

	virtual void setScale(glm::vec3 scale) = 0;

	virtual void setColor(glm::vec3 color) = 0;

	virtual void setLightColor(glm::vec3 color) = 0;

	virtual void setLightPosition(glm::vec3 position) = 0;

	virtual void setVertexSource(std::string vertexShader) = 0;
	
	virtual void setFragmentSource(std::string fragShader) = 0;

protected:
	glm::vec3 mPosition;
	glm::vec3 mScale;

	// Vertex buffers.
	GLuint mVao;
	GLuint mVbo;
	GLuint mIbo;

	GLuint mNumIndices;

	// Shader data.
	GLuint mVertHandle;
	GLuint mFragHandle;
	GLuint mProgramHandle;
	glx::ShaderFile vertexSource;
	glx::ShaderFile fragmentSource;

	// Color data
	glm::vec3 mObjectColor;
	glm::vec3 mLightColor;
	glm::vec3 mLightPosition;

	// Uniform variable data.
	GLuint mUniformModelLoc;
	GLuint mUniformViewLoc;
	GLuint mUniformProjectionLoc;
	GLuint mUniformObjectColorLoc;
	GLuint mUniformLightColorLoc;
	GLuint mUniformLightPosition;
	GLuint mUniformViewPosition;

	// Option Switches
	bool mSpecular;
};

class Triangle : public SceneObject
{
public:
	Triangle::Triangle()
	{
		// allocate the memory to hold the program and shader data
		mProgramHandle = glCreateProgram();
		mVertHandle = glCreateShader(GL_VERTEX_SHADER);
		mFragHandle = glCreateShader(GL_FRAGMENT_SHADER);
	}

	void Triangle::loadShaders(std::string vertexShader, std::string fragShader)
	{
		std::string shaderRoot{ ShaderPath };
		vertexSource =
			glx::readShaderSource(shaderRoot + "triangle.vert", IncludeDir);
		fragmentSource =
			glx::readShaderSource(shaderRoot + "triangle.frag", IncludeDir);

		if (auto result{ glx::compileShader(vertexSource.sourceString, mVertHandle) };
			result)
		{
			throw OpenGLError(*result);
		}

		if (auto result =
			glx::compileShader(fragmentSource.sourceString, mFragHandle);
			result)
		{
			throw OpenGLError(*result);
		}

		// communicate to OpenGL the shaders used to render the Triangle
		glAttachShader(mProgramHandle, mVertHandle);
		glAttachShader(mProgramHandle, mFragHandle);

		if (auto result = glx::linkShaders(mProgramHandle); result)
		{
			throw OpenGLError(*result);
		}

		setupUniformVariables();
	}

	void Triangle::loadDataToGPU(std::vector<float> const& vertices, [[maybe_unused]] std::vector<GLuint> const& indices)
	{
		// create buffer to hold triangle vertex data
		glCreateBuffers(1, &mVbo);
		// allocate and initialize buffer to vertex data
		glNamedBufferStorage(
			mVbo, glx::size<float>(vertices.size()), vertices.data(), 0);

		// create holder for all buffers
		glCreateVertexArrays(1, &mVao);
		// bind vertex buffer to the vertex array
		glVertexArrayVertexBuffer(mVao, 0, mVbo, 0, glx::stride<float>(6));

		// enable attributes for the two components of a vertex
		glEnableVertexArrayAttrib(mVao, 0);
		glEnableVertexArrayAttrib(mVao, 1);

		// specify to OpenGL how the vertices and colors are laid out in the buffer
		glVertexArrayAttribFormat(
			mVao, 0, 3, GL_FLOAT, GL_FALSE, glx::relativeOffset<float>(0));
		glVertexArrayAttribFormat(
			mVao, 1, 3, GL_FLOAT, GL_FALSE, glx::relativeOffset<float>(3));

		// associate the vertex attributes (coordinates and color) to the vertex
		// attribute
		glVertexArrayAttribBinding(mVao, 0, 0);
		glVertexArrayAttribBinding(mVao, 1, 0);
	}

	void Triangle::reloadShaders()
	{
		if (glx::shouldShaderBeReloaded(vertexSource))
		{
			glx::reloadShader(
				mProgramHandle, mVertHandle, vertexSource, IncludeDir);
		}

		if (glx::shouldShaderBeReloaded(fragmentSource))
		{
			glx::reloadShader(
				mProgramHandle, mFragHandle, fragmentSource, IncludeDir);
		}
	}

	void Triangle::loadNewShaders()
	{
		glx::reloadShader(
			mProgramHandle, mVertHandle, vertexSource, IncludeDir);
		glx::reloadShader(
			mProgramHandle, mFragHandle, fragmentSource, IncludeDir);
		setupUniformVariables();
	}

	void Triangle::render([[maybe_unused]] bool paused,
		[[maybe_unused]] int width,
		[[maybe_unused]] int height,
		[[maybe_unused]] Camera *camera)
	{
		reloadShaders();

		// tell OpenGL which program object to use to render the Triangle
		glUseProgram(mProgramHandle);

		// tell OpenGL which vertex array object to use to render the Triangle
		glBindVertexArray(mVao);
		// actually render the Triangle
		glDrawArrays(GL_TRIANGLES, 0, 36);
	}

	void Triangle::setVertexSource(std::string vertexShader)
	{
		std::string shaderRoot{ ShaderPath };
		vertexSource =
			glx::readShaderSource(shaderRoot + vertexShader, IncludeDir);
	}

	void Triangle::setFragmentSource(std::string fragShader)
	{
		std::string shaderRoot{ ShaderPath };
		fragmentSource =
			glx::readShaderSource(shaderRoot + fragShader, IncludeDir);
	}

	void Triangle::freeGPUData()
	{
		// unwind all the allocations made
		glDeleteVertexArrays(1, &mVao);
		glDeleteBuffers(1, &mVbo);
		glDeleteShader(mFragHandle);
		glDeleteShader(mVertHandle);
		glDeleteProgram(mProgramHandle);
	}
private:
	void Triangle::setupUniformVariables()
	{
		mUniformModelLoc = glGetUniformLocation(mProgramHandle, "model");
		mUniformViewLoc = glGetUniformLocation(mProgramHandle, "view");;
		mUniformProjectionLoc = glGetUniformLocation(mProgramHandle, "projection");
	}
};

class Cube : public SceneObject
{
public:
	Cube::Cube()
	{
		// allocate the memory to hold the program and shader data
		mProgramHandle = glCreateProgram();
		mVertHandle = glCreateShader(GL_VERTEX_SHADER);
		mFragHandle = glCreateShader(GL_FRAGMENT_SHADER);
	}

	void Cube::loadShaders(std::string vertexShader, std::string fragShader)
	{
		std::string shaderRoot{ ShaderPath };
		vertexSource =
			glx::readShaderSource(shaderRoot + vertexShader, IncludeDir);
		fragmentSource =
			glx::readShaderSource(shaderRoot + fragShader, IncludeDir);

		if (auto result{ glx::compileShader(vertexSource.sourceString, mVertHandle) };
			result)
		{
			throw OpenGLError(*result);
		}

		if (auto result =
			glx::compileShader(fragmentSource.sourceString, mFragHandle);
			result)
		{
			throw OpenGLError(*result);
		}

		// communicate to OpenGL the shaders used to render the Triangle
		glAttachShader(mProgramHandle, mVertHandle);
		glAttachShader(mProgramHandle, mFragHandle);

		if (auto result = glx::linkShaders(mProgramHandle); result)
		{
			throw OpenGLError(*result);
		}

		setupUniformVariables();
	}


	void Cube::loadDataToGPU(std::vector<float> const& vertices, [[maybe_unused]] std::vector<GLuint> const& indices)
	{
		// create buffer to hold triangle vertex data
		glCreateBuffers(1, &mVbo);
		// allocate and initialize buffer to vertex data
		glNamedBufferStorage(
			mVbo, glx::size<float>(vertices.size()), vertices.data(), 0);

		// create holder for all buffers
		glCreateVertexArrays(1, &mVao);
		// bind vertex buffer to the vertex array
		glVertexArrayVertexBuffer(mVao, 0, mVbo, 0, glx::stride<float>(6));

		// enable attributes for the two components of a vertex
		glEnableVertexArrayAttrib(mVao, 0);
		glEnableVertexArrayAttrib(mVao, 1);

		// specify to OpenGL how the vertices and colors are laid out in the buffer
		glVertexArrayAttribFormat(
			mVao, 0, 3, GL_FLOAT, GL_FALSE, glx::relativeOffset<float>(0));
		// add normal attributes
		glVertexArrayAttribFormat(
			mVao, 1, 3, GL_FLOAT, GL_FALSE, glx::relativeOffset<float>(3));

		// associate the vertex attributes (coordinates and color) to the vertex
		// attribute
		glVertexArrayAttribBinding(mVao, 0, 0);
		glVertexArrayAttribBinding(mVao, 1, 0);
	}

	void Cube::reloadShaders()
	{
		if (glx::shouldShaderBeReloaded(vertexSource))
		{
			glx::reloadShader(
				mProgramHandle, mVertHandle, vertexSource, IncludeDir);
		}

		if (glx::shouldShaderBeReloaded(fragmentSource))
		{
			glx::reloadShader(
				mProgramHandle, mFragHandle, fragmentSource, IncludeDir);
		}
	}

	void Cube::loadNewShaders()
	{
		glx::reloadShader(
			mProgramHandle, mVertHandle, vertexSource, IncludeDir);
		glx::reloadShader(
			mProgramHandle, mFragHandle, fragmentSource, IncludeDir);
		setupUniformVariables();
	}

	void Cube::render([[maybe_unused]] bool paused,
		[[maybe_unused]] int width,
		[[maybe_unused]] int height,
		[[maybe_unused]] Camera *camera)
	{
		reloadShaders();

		// tell OpenGL which program object to use to render the Triangle
		glUseProgram(mProgramHandle);

		glm::mat4 model = glm::mat4(1.0f);
		glm::mat4 view = glm::mat4(1.0f);
		glm::mat4 projection = glm::mat4(1.0f);
		model = glm::translate(model, glm::vec3(5.0f, 0.0f, 0.0f));
		model = glm::rotate(model, glm::radians(15.0f * (float)glfwGetTime()), glm::vec3(1.0f, 0.0f, 0.0f));
		view = camera->GetViewMatrix();
		projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);

		glUniformMatrix4fv(mUniformModelLoc, 1, GL_FALSE, glm::value_ptr(model));
		glUniformMatrix4fv(mUniformViewLoc, 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(mUniformProjectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
		glUniform3fv(mUniformObjectColorLoc, 1, glm::value_ptr(mObjectColor));
		glUniform3fv(mUniformLightColorLoc, 1, glm::value_ptr(mLightColor));
		glUniform3fv(mUniformLightPosition, 1, glm::value_ptr(mLightPosition));
		glUniform3fv(mUniformViewPosition, 1, glm::value_ptr(camera->mPosition));

		// tell OpenGL which vertex array object to use to render the Triangle
		glBindVertexArray(mVao);
		// actually render the Triangle
		glDrawArrays(GL_TRIANGLES, 0, 36);
	}

	void Cube::freeGPUData()
	{
		// unwind all the allocations made
		glDeleteVertexArrays(1, &mVao);
		glDeleteBuffers(1, &mVbo);
		glDeleteShader(mFragHandle);
		glDeleteShader(mVertHandle);
		glDeleteProgram(mProgramHandle);
	}

	void Cube::setPosition(glm::vec3 position)
	{
		mPosition = position;
	}

	void Cube::setScale(glm::vec3 scale)
	{
		mScale = scale;
	}

	void Cube::setColor(glm::vec3 color)
	{
		mObjectColor = color;
	}

	void Cube::setLightColor(glm::vec3 color)
	{
		mLightColor = color;
	}

	void Cube::setLightPosition(glm::vec3 position)
	{
		mLightPosition = position;
	}

	void Cube::setVertexSource(std::string vertexShader)
	{
		std::string shaderRoot{ ShaderPath };
		vertexSource =
			glx::readShaderSource(shaderRoot + vertexShader, IncludeDir);
	}

	void Cube::setFragmentSource(std::string fragShader)
	{
		std::string shaderRoot{ ShaderPath };
		fragmentSource =
			glx::readShaderSource(shaderRoot + fragShader, IncludeDir);
	}

protected:
	void Cube::setupUniformVariables()
	{
		mUniformModelLoc = glGetUniformLocation(mProgramHandle, "model");
		mUniformViewLoc = glGetUniformLocation(mProgramHandle, "view");;
		mUniformProjectionLoc = glGetUniformLocation(mProgramHandle, "projection");;
		mUniformObjectColorLoc = glGetUniformLocation(mProgramHandle, "objectColor");
		mUniformLightColorLoc = glGetUniformLocation(mProgramHandle, "lightColor");
		mUniformLightPosition = glGetUniformLocation(mProgramHandle, "lightPos");
		mUniformViewPosition = glGetUniformLocation(mProgramHandle, "viewPos");
	}
};

class MeshObject : public SceneObject
{
public:
	MeshObject::MeshObject()
	{
		// allocate the memory to hold the program and shader data
		mProgramHandle = glCreateProgram();
		mVertHandle = glCreateShader(GL_VERTEX_SHADER);
		mFragHandle = glCreateShader(GL_FRAGMENT_SHADER);
	}

	void MeshObject::loadShaders(std::string vertexShader, std::string fragShader)
	{
		std::string shaderRoot{ ShaderPath };
		vertexSource =
			glx::readShaderSource(shaderRoot + vertexShader, IncludeDir);
		fragmentSource =
			glx::readShaderSource(shaderRoot + fragShader, IncludeDir);

		if (auto result{ glx::compileShader(vertexSource.sourceString, mVertHandle) };
			result)
		{
			throw OpenGLError(*result);
		}

		if (auto result =
			glx::compileShader(fragmentSource.sourceString, mFragHandle);
			result)
		{
			throw OpenGLError(*result);
		}

		// communicate to OpenGL the shaders used to render the Triangle
		glAttachShader(mProgramHandle, mVertHandle);
		glAttachShader(mProgramHandle, mFragHandle);

		if (auto result = glx::linkShaders(mProgramHandle); result)
		{
			throw OpenGLError(*result);
		}

		setupUniformVariables();
	}

	void MeshObject::loadDataToGPU(std::vector<float> const& vertices, [[maybe_unused]] std::vector<GLuint> const& indices)
	{
		mNumIndices = static_cast<GLsizei>(indices.size());

		// create buffer to hold triangle vertex data
		glCreateBuffers(1, &mVbo);
		// allocate and initialize buffer to vertex data
		glNamedBufferStorage(
			mVbo, glx::size<float>(vertices.size()), vertices.data(), 0);

		glCreateBuffers(1, &mIbo);
		glNamedBufferStorage(
			mIbo, glx::size<GLuint>(indices.size()), indices.data(), 0);

		// create holder for all buffers
		glCreateVertexArrays(1, &mVao);
		// bind vertex buffer to the vertex array
		glVertexArrayVertexBuffer(mVao, 0, mVbo, 0, glx::stride<float>(6));

		glVertexArrayElementBuffer(mVao, mIbo);

		// enable attributes for the two components of a vertex
		glEnableVertexArrayAttrib(mVao, 0);
		glEnableVertexArrayAttrib(mVao, 1);


		glEnableVertexArrayAttrib(mVao, 0);
		glEnableVertexArrayAttrib(mVao, 1);

		glVertexArrayAttribFormat(mVao, 0, 3, GL_FLOAT,
			GL_FALSE, glx::relativeOffset<float>(0));
		glVertexArrayAttribFormat(mVao, 1, 3, GL_FLOAT,
			GL_FALSE, glx::relativeOffset<float>(3));

		glVertexArrayAttribBinding(mVao, 0, 0);
		glVertexArrayAttribBinding(mVao, 1, 0);
	}

	void MeshObject::reloadShaders()
	{
		if (glx::shouldShaderBeReloaded(vertexSource))
		{
			glx::reloadShader(
				mProgramHandle, mVertHandle, vertexSource, IncludeDir);
		}

		if (glx::shouldShaderBeReloaded(fragmentSource))
		{
			glx::reloadShader(
				mProgramHandle, mFragHandle, fragmentSource, IncludeDir);
		}
	}

	void MeshObject::loadNewShaders()
	{
		glx::reloadShader(
			mProgramHandle, mVertHandle, vertexSource, IncludeDir);
		glx::reloadShader(
			mProgramHandle, mFragHandle, fragmentSource, IncludeDir);
		setupUniformVariables();
	}

	void MeshObject::render([[maybe_unused]] bool paused,
		[[maybe_unused]] int width,
		[[maybe_unused]] int height,
		[[maybe_unused]] Camera* camera)
	{
		reloadShaders();

		// tell OpenGL which program object to use to render the Triangle
		glUseProgram(mProgramHandle);

		glm::mat4 model = glm::mat4(1.0f);
		glm::mat4 view = glm::mat4(1.0f);
		glm::mat4 projection = glm::mat4(1.0f);
		model = glm::translate(model, mPosition);
		model = glm::rotate(model, glm::radians(15.0f) * (float)glfwGetTime(), glm::vec3(0.0f, 1.0f, 0.0f));
		view = camera->GetViewMatrix();
		projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);

		glUniformMatrix4fv(mUniformModelLoc, 1, GL_FALSE, glm::value_ptr(model));
		glUniformMatrix4fv(mUniformViewLoc, 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(mUniformProjectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
		glUniform3fv(mUniformObjectColorLoc, 1, glm::value_ptr(mObjectColor));
		glUniform3fv(mUniformLightColorLoc, 1, glm::value_ptr(mLightColor));
		glUniform3fv(mUniformLightPosition, 1, glm::value_ptr(mLightPosition));
		glUniform3fv(mUniformViewPosition, 1, glm::value_ptr(camera->mPosition));

		glUseProgram(mProgramHandle);
		// tell OpenGL which vertex array object to use to render the Triangle
		glBindVertexArray(mVao);
		// actually render the Triangle
		glDrawElements(GL_TRIANGLES, mNumIndices, GL_UNSIGNED_INT,
						glx::bufferOffset<GLuint>(0));
	}

	void MeshObject::freeGPUData()
	{
		// unwind all the allocations made
		glDeleteVertexArrays(1, &mVao);
		glDeleteBuffers(1, &mVbo);
		glDeleteShader(mFragHandle);
		glDeleteShader(mVertHandle);
		glDeleteProgram(mProgramHandle);
	}

	void MeshObject::setPosition(glm::vec3 position)
	{
		mPosition = position;
	}

	void MeshObject::setScale(glm::vec3 scale)
	{
		mScale = scale;
	}

	void MeshObject::setColor(glm::vec3 color)
	{
		mObjectColor = color;
	}

	void MeshObject::setLightColor(glm::vec3 color)
	{
		mLightColor = color;
	}

	void MeshObject::setLightPosition(glm::vec3 position)
	{
		mLightPosition = position;
	}

	void MeshObject::setVertexSource(std::string vertexShader)
	{
		std::string shaderRoot{ ShaderPath };
		vertexSource =
			glx::readShaderSource(shaderRoot + vertexShader, IncludeDir);
	}

	void MeshObject::setFragmentSource(std::string fragShader)
	{
		std::string shaderRoot{ ShaderPath };
		fragmentSource =
			glx::readShaderSource(shaderRoot + fragShader, IncludeDir);
	}

protected:
	void MeshObject::setupUniformVariables()
	{
		mUniformModelLoc = glGetUniformLocation(mProgramHandle, "model");
		mUniformViewLoc = glGetUniformLocation(mProgramHandle, "view");;
		mUniformProjectionLoc = glGetUniformLocation(mProgramHandle, "projection");;
		mUniformObjectColorLoc = glGetUniformLocation(mProgramHandle, "objectColor");
		mUniformLightColorLoc = glGetUniformLocation(mProgramHandle, "lightColor");
		mUniformLightPosition = glGetUniformLocation(mProgramHandle, "lightPos");
		mUniformViewPosition = glGetUniformLocation(mProgramHandle, "viewPos");
	}
};

class PointLight : public SceneObject
{
public:

	PointLight::PointLight()
	{
		// allocate the memory to hold the program and shader data
		mProgramHandle = glCreateProgram();
		mVertHandle = glCreateShader(GL_VERTEX_SHADER);
		mFragHandle = glCreateShader(GL_FRAGMENT_SHADER);
	}

	void PointLight::loadShaders(std::string vertexShader, std::string fragShader)
	{
		std::string shaderRoot{ ShaderPath };
		vertexSource =
			glx::readShaderSource(shaderRoot + "PointLight.vert", IncludeDir);
		fragmentSource =
			glx::readShaderSource(shaderRoot + "PointLight.frag", IncludeDir);

		if (auto result{ glx::compileShader(vertexSource.sourceString, mVertHandle) };
			result)
		{
			throw OpenGLError(*result);
		}

		if (auto result =
			glx::compileShader(fragmentSource.sourceString, mFragHandle);
			result)
		{
			throw OpenGLError(*result);
		}

		// communicate to OpenGL the shaders used to render the Triangle
		glAttachShader(mProgramHandle, mVertHandle);
		glAttachShader(mProgramHandle, mFragHandle);

		if (auto result = glx::linkShaders(mProgramHandle); result)
		{
			throw OpenGLError(*result);
		}

		setupUniformVariables();
	}

	void PointLight::loadDataToGPU(std::vector<float> const& vertices, [[maybe_unused]] std::vector<GLuint> const& indices)
	{
		// create buffer to hold triangle vertex data
		glCreateBuffers(1, &mVbo);
		// allocate and initialize buffer to vertex data
		glNamedBufferStorage(
			mVbo, glx::size<float>(vertices.size()), vertices.data(), 0);

		// create holder for all buffers
		glCreateVertexArrays(1, &mVao);
		// bind vertex buffer to the vertex array
		glVertexArrayVertexBuffer(mVao, 0, mVbo, 0, glx::stride<float>(6));

		// enable attributes for the two components of a vertex
		glEnableVertexArrayAttrib(mVao, 0);
		glEnableVertexArrayAttrib(mVao, 1);

		// specify to OpenGL how the vertices and colors are laid out in the buffer
		glVertexArrayAttribFormat(
			mVao, 0, 3, GL_FLOAT, GL_FALSE, glx::relativeOffset<float>(0));
		glVertexArrayAttribFormat(
			mVao, 1, 3, GL_FLOAT, GL_FALSE, glx::relativeOffset<float>(3));

		// associate the vertex attributes (coordinates and color) to the vertex
		// attribute
		glVertexArrayAttribBinding(mVao, 0, 0);
		glVertexArrayAttribBinding(mVao, 1, 0);
	}

	void PointLight::reloadShaders()
	{
		if (glx::shouldShaderBeReloaded(vertexSource))
		{
			glx::reloadShader(
				mProgramHandle, mVertHandle, vertexSource, IncludeDir);
		}

		if (glx::shouldShaderBeReloaded(fragmentSource))
		{
			glx::reloadShader(
				mProgramHandle, mFragHandle, fragmentSource, IncludeDir);
		}
		setupUniformVariables();
	}

	void PointLight::loadNewShaders()
	{
		glx::reloadShader(
			mProgramHandle, mVertHandle, vertexSource, IncludeDir);
		glx::reloadShader(
			mProgramHandle, mFragHandle, fragmentSource, IncludeDir);
		setupUniformVariables();
	}

	void PointLight::render([[maybe_unused]] bool paused,
		[[maybe_unused]] int width,
		[[maybe_unused]] int height,
		[[maybe_unused]] Camera* camera)
	{
		reloadShaders();

		// tell OpenGL which program object to use to render the Triangle
		glUseProgram(mProgramHandle);

		glm::mat4 model = glm::mat4(1.0f);
		glm::mat4 view = glm::mat4(1.0f);
		glm::mat4 projection = glm::mat4(1.0f);
		model = glm::translate(model, mPosition);
		model = glm::scale(model, glm::vec3(0.2f));
		view = camera->GetViewMatrix();
		projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);

		glUniformMatrix4fv(mUniformModelLoc, 1, GL_FALSE, glm::value_ptr(model));
		glUniformMatrix4fv(mUniformViewLoc, 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(mUniformProjectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
		glUniform3fv(mUniformObjectColorLoc, 1, glm::value_ptr(mObjectColor));

		// tell OpenGL which vertex array object to use to render the Triangle
		glBindVertexArray(mVao);
		// actually render the Triangle
		glDrawArrays(GL_TRIANGLES, 0, 36);
	}

	void PointLight::freeGPUData()
	{
		// unwind all the allocations made
		glDeleteVertexArrays(1, &mVao);
		glDeleteBuffers(1, &mVbo);
		glDeleteShader(mFragHandle);
		glDeleteShader(mVertHandle);
		glDeleteProgram(mProgramHandle);
	}

	void PointLight::setPosition(glm::vec3 position)
	{
		mPosition = position;
	}

	void PointLight::setScale(glm::vec3 scale)
	{
		mScale = scale;
	}

	void PointLight::setColor(glm::vec3 color)
	{
		mObjectColor = color;
	}

	void PointLight::setLightColor(glm::vec3 color)
	{
		mLightColor = color;
	}

	void PointLight::setLightPosition(glm::vec3 position)
	{
		mLightPosition = position;
	}

	void PointLight::setVertexSource(std::string vertexShader)
	{
		std::string shaderRoot{ ShaderPath };
		vertexSource =
			glx::readShaderSource(shaderRoot + vertexShader, IncludeDir);
	}

	void PointLight::setFragmentSource(std::string fragShader)
	{
		std::string shaderRoot{ ShaderPath };
		fragmentSource =
			glx::readShaderSource(shaderRoot + fragShader, IncludeDir);
	}

protected:
	void PointLight::setupUniformVariables()
	{
		mUniformModelLoc = glGetUniformLocation(mProgramHandle, "model");
		mUniformViewLoc = glGetUniformLocation(mProgramHandle, "view");;
		mUniformProjectionLoc = glGetUniformLocation(mProgramHandle, "projection");;
		mUniformObjectColorLoc = glGetUniformLocation(mProgramHandle, "objectColor");
		mUniformLightColorLoc = glGetUniformLocation(mProgramHandle, "lightColor");
	}
};


class Program
{
public:
    Program(int width, int height, std::string title);

    void run(std::vector<std::shared_ptr<SceneObject>> scene);

	int loadMesh(const std::string& meshPath, std::vector<GLuint> &indices, std::vector<float> &vertices);

    void freeGPUData();

	void Program::processInput(GLFWwindow* window, Camera *camera);

	bool wireframe = false;

private:
    static void errorCallback(int code, char const* message)
    {
        fmt::print("error ({}): {}\n", code, message);
    }

    void createGLContext();

	float mLastX;
	float mLastY;
	bool mFirstMouse;

	float mDeltaTime;
	float mLastFrame;

	Camera mCamera;

    GLFWwindow* mWindow;
    glx::WindowSettings settings;
    glx::WindowCallbacks callbacks;

    bool paused;
	bool renderObject0;
	bool renderObject1;
	bool renderObject2;
	bool renderObject3;

	bool specularShading0;
	bool specularShading1;
	bool specularShading2;
	bool specularShading3;
};
