#include "assignment.hpp"

// ===------------IMPLEMENTATIONS-------------===

void Program::processInput(GLFWwindow* window, Camera *camera)
{
	const float cameraSpeed = 10.0f * mDeltaTime;
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera->ProcessKeyboard(FORWARD, mDeltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera->ProcessKeyboard(LEFT, mDeltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera->ProcessKeyboard(BACKWARD, mDeltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera->ProcessKeyboard(RIGHT, mDeltaTime);
}

Program::Program(int width, int height, std::string title) :
    settings{}, callbacks{}, paused{}, mWindow{nullptr}
{
    settings.size.width  = width;
    settings.size.height = height;
    settings.title       = title;
	settings.isFullscreen = false;

    if (!glx::initializeGLFW(errorCallback))
    {
        throw OpenGLError("Failed to initialize GLFW with error callback");
    }

    mWindow = glx::createGLFWWindow(settings);
    if (mWindow == nullptr)
    {
        throw OpenGLError("Failed to create GLFW Window");
    }

	// camera
	Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
	mCamera = camera;
	mLastX = width / 2.0f;
	mLastY = height / 2.0f;

	// timing
	mDeltaTime = 0.0f;	// Time between current frame and last frame
	mLastFrame = 0.0f;	// Time of last frame

	// Mouse input
	callbacks.mouseMoveCallback = [&](double xPosition, double yPosition) {
		if (mFirstMouse)
		{
			mLastX = (float)xPosition;
			mLastY = (float)yPosition;
			mFirstMouse = false;
		}

		float xoffset = (float)xPosition - mLastX;
		float yoffset = mLastY - (float)yPosition;	// reversed since y-coordinates go from bottom to top
		mLastX = (float)xPosition;
		mLastY = (float)yPosition;

		mCamera.ProcessMouseMovement(xoffset, yoffset);
	};


	callbacks.keyPressCallback = [&](int key, int, int action, int) {
		if (key == GLFW_KEY_SPACE && action == GLFW_RELEASE)
		{
			wireframe = !wireframe;
		}
		if (key == GLFW_KEY_KP_0 && action == GLFW_RELEASE)
		{
			renderObject0 = !renderObject0;
		}
		if (key == GLFW_KEY_KP_1 && action == GLFW_RELEASE)
		{
			renderObject1 = !renderObject1;
		}
		if (key == GLFW_KEY_KP_2 && action == GLFW_RELEASE)
		{
			renderObject2 = !renderObject2;
		}
		if (key == GLFW_KEY_KP_3 && action == GLFW_RELEASE)
		{
			renderObject3 = !renderObject3;
		}
		if (key == GLFW_KEY_0 && action == GLFW_RELEASE)
		{
			specularShading0 = !specularShading0;
		}
		if (key == GLFW_KEY_1 && action == GLFW_RELEASE)
		{
			specularShading1 = !specularShading1;
		}
		if (key == GLFW_KEY_2 && action == GLFW_RELEASE)
		{
			specularShading2 = !specularShading2;
		}
		if (key == GLFW_KEY_3 && action == GLFW_RELEASE)
		{
			specularShading3 = !specularShading3;
		}
	};

	glfwSetInputMode(mWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    createGLContext();
}

//Pass in a list of items to render
void Program::run(std::vector<std::shared_ptr<SceneObject>> scene)
{
    glEnable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(mWindow))
    {
        int width;
        int height;

		float currentFrame = (float)glfwGetTime();
		mDeltaTime = currentFrame - mLastFrame;
		mLastFrame = currentFrame;

		processInput(mWindow, &mCamera);

        glfwGetFramebufferSize(mWindow, &width, &height);
        // setup the view to be the window's size
        glViewport(0, 0, width, height);
        // tell OpenGL the what color to clear the screen to
        glClearColor(0, 0, 0, 1);
        // actually clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		/*
		for (auto item : scene)
		{
			item->render(paused, width, height, &mCamera);
		}
		*/

		for (auto it = scene.begin(); it != scene.end(); it++)
		{
			int index = (int)std::distance(scene.begin(), it);
			// Rendering objects on/off
			if(index == 0 && renderObject0)
				scene[0]->render(paused, width, height, &mCamera);
			if (index == 1 && renderObject1)
				scene[1]->render(paused, width, height, &mCamera);
			if (index == 2 && renderObject2)
				scene[2]->render(paused, width, height, &mCamera);
			if (index == 3 && renderObject3)
				scene[3]->render(paused, width, height, &mCamera);
			// Specular shading on/off
			if (index == 0 && !specularShading0) {
				scene[0]->setVertexSource("diffuse.vert");
				scene[0]->setFragmentSource("diffuse.frag");
				scene[0]->loadNewShaders();
			}
			if (index == 0 && specularShading0) {
				scene[0]->setVertexSource("specular.vert");
				scene[0]->setFragmentSource("specular.frag");
				scene[0]->loadNewShaders();
			}
			if (index == 1 && !specularShading1) {
				scene[1]->setVertexSource("diffuse.vert");
				scene[1]->setFragmentSource("diffuse.frag");
				scene[1]->loadNewShaders();
			}
			if (index == 1 && specularShading1) {
				scene[1]->setVertexSource("specular.vert");
				scene[1]->setFragmentSource("specular.frag");
				scene[1]->loadNewShaders();
			}
			if (index == 2 && !specularShading2) {
				scene[2]->setVertexSource("diffuse.vert");
				scene[2]->setFragmentSource("diffuse.frag");
				scene[2]->loadNewShaders();
			}
			if (index == 2 && specularShading2) {
				scene[2]->setVertexSource("specular.vert");
				scene[2]->setFragmentSource("specular.frag");
				scene[2]->loadNewShaders();
			}
			if (index == 2 && !specularShading2) {
				scene[3]->setVertexSource("PointLight.vert");
				scene[3]->setFragmentSource("PointLight.frag");
				scene[3]->loadNewShaders();
			}
			if (index == 2 && specularShading2) {
				scene[3]->setVertexSource("PointLight.vert");
				scene[3]->setFragmentSource("PointLight.frag");
				scene[3]->loadNewShaders();
			}
		}

		if (wireframe == true)
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		else
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        glfwSwapBuffers(mWindow);
        glfwPollEvents();
    }
}
  
void Program::freeGPUData()
{
    glx::destroyGLFWWindow(mWindow);
    glx::terminateGLFW();
}

void Program::createGLContext()
{
    using namespace magic_enum::bitwise_operators;

    glx::bindWindowCallbacks(mWindow, callbacks);
    glfwMakeContextCurrent(mWindow);
    glfwSwapInterval(1);

    if (!glx::createGLContext(mWindow, settings.version))
    {
        throw OpenGLError("Failed to create OpenGL context");
    }

    glx::initializeGLCallback(glx::ErrorSource::All,
                              glx::ErrorType::All,
                              glx::ErrorSeverity::High |
                                  glx::ErrorSeverity::Medium);
}

int Program::loadMesh(const std::string &meshPath, std::vector<GLuint> &indices, std::vector<float> &vertices)
{
	GLsizei count;
	auto result = utils::loadObjMesh(meshPath);
	if (!result)
	{
		return 0; // Something went wrong
	}

	utils::ObjMesh mesh = result.value();

	for(auto it = mesh.shapes.begin(); it != mesh.shapes.end(); it++)
	{
		auto index = std::distance(mesh.shapes.begin(), it);
		count = static_cast<GLsizei>(mesh.shapes[index].indices.size());
		indices.resize(count);
		std::transform(
			mesh.shapes[index].indices.begin(), mesh.shapes[index].indices.end(),
			indices.begin(),
			[](std::size_t i) -> GLuint { return static_cast<GLuint>(i); });

		//std::vector<float> vertices;
		for (auto& vertex : mesh.shapes[index].vertices)
		{
			vertices.push_back(vertex.position.x);
			vertices.push_back(vertex.position.y);
			vertices.push_back(vertex.position.z);

			vertices.push_back(vertex.normal.x);
			vertices.push_back(vertex.normal.y);
			vertices.push_back(vertex.normal.z);
		}
	}

	return -1;
}

// ===-----------------DRIVER-----------------===

int main()
{
	try
	{
		// clang-format off
		std::vector<float> vertices
		{
			// Vertices          Colours
			0.4f, -0.4f, 0.0f,   1.0f, 0.0f, 0.0f,
		   -0.4f, -0.4f, 0.0f,   0.0f, 1.0f, 0.0f,
			0.0f,  0.4f, 0.0f,   0.0f, 0.0f, 1.0f
		};
		// clang-format on

		// clang-format off
		std::vector<float> cube_vertices = {
			-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
			 0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
			 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
			 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
			-0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
			-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,

			-0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
			 0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
			 0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
			 0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
			-0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
			-0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,

			-0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
			-0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
			-0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
			-0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
			-0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
			-0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,

			 0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
			 0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
			 0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
			 0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
			 0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
			 0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,

			-0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
			 0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
			 0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
			 0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
			-0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
			-0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,

			-0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
			 0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
			 0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
			 0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
			-0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
			-0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f
		};
		// clang-format off

		// Used as argument for scene objects with no index buffer object
		std::vector<GLuint> empty_indices;

		Program prog{ 1920, 1080, "CSC305 Lab 6" };

		//Load objects into the scene
		std::vector<std::shared_ptr<SceneObject>> scene;
		glm::vec3 lightPos{ glm::vec3{1.2f, 1.0f, 2.0f} };

		// Load cube
		scene.push_back(std::make_shared<Cube>());
		scene[0]->loadShaders("specular.vert", "specular.frag");
		scene[0]->loadDataToGPU(cube_vertices, empty_indices);
		scene[0]->setColor(glm::vec4{ 1.0, 0.0, 0.0, 1.0 });
		scene[0]->setLightColor(glm::vec3{ 1.0, 1.0, 1.0 });
		scene[0]->setLightPosition(lightPos);

		// Load Suzanne
		std::string SuzannePath{ ShaderPath };
		std::string suzanne{ "suzanne.obj" };
		SuzannePath = SuzannePath + suzanne;
		std::vector<GLuint> suzanneIndices;
		std::vector<float> suzanneVertices;
		if (!prog.loadMesh(SuzannePath, suzanneIndices, suzanneVertices))
		{
			return -1;
		}

		scene.push_back(std::make_shared<MeshObject>());
		scene[1]->loadShaders("specular.vert", "specular.frag");
		scene[1]->loadDataToGPU(suzanneVertices, suzanneIndices);
		scene[1]->setColor(glm::vec4{ 0.949, 0.964, 0.772, 1.0 });
		scene[1]->setLightColor(glm::vec3{ 1.0, 1.0, 1.0 });
		scene[1]->setLightPosition(lightPos);
		scene[1]->setPosition({ 0.0f, 0.0f, 0.0f });

		// Load Bunny
		std::string BunnyPath{ ShaderPath };
		std::string bunny{ "bunny.obj" };
		BunnyPath = BunnyPath + bunny;
		std::vector<GLuint> bunnyIndices;
		std::vector<float> bunnyVertices;
		if (!prog.loadMesh(BunnyPath, bunnyIndices, bunnyVertices))
		{
			return -1;
		}

		scene.push_back(std::make_shared<MeshObject>());
		scene[2]->loadShaders("specular.vert", "specular.frag");
		scene[2]->loadDataToGPU(bunnyVertices, bunnyIndices);
		scene[2]->setColor(glm::vec4{ 0.729, 0.560, 0.427, 1.0 });
		scene[2]->setLightColor(glm::vec3{ 1.0, 1.0, 1.0 });
		scene[2]->setLightPosition(lightPos);
		scene[2]->setPosition({ -4.0f, 0.0f, 0.0f });

		// Load point light
		scene.push_back(std::make_shared<PointLight>());
		scene[3]->loadShaders("specular.vert", "specular.frag");
		scene[3]->loadDataToGPU(cube_vertices, empty_indices);
		scene[3]->setPosition(lightPos);
		scene[3]->setScale(glm::vec3(0.2f));
		scene[3]->setColor(glm::vec3{ 1.0, 1.0, 1.0 });		

		prog.run(scene);

        prog.freeGPUData();
		for (auto item : scene)
		{
			item->freeGPUData();
		}
    }
    catch (OpenGLError& err)
    {
        fmt::print("OpenGL Error:\n\t{}\n", err.what());
    }

    return 0;
}
