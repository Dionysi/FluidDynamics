#include "stdfax.h"
#include "Application.h"
#include "Game.h"
#include <chrono>


// Initialize static member-variables. 
uint Application::s_RenderWidth = 0;
uint Application::s_RenderHeight = 0;
uint Application::s_WindowWidth = 0;
uint Application::s_WindowHeight = 0;

bool Application::s_Initialized = false;
GLFWwindow* Application::s_Window = nullptr;
Surface* Application::s_RenderSurface = nullptr;
clContext* Application::s_clContext = nullptr;

int main() {
	Application::Initialize(1024, 1024);
	Application::Run();

	return 0;
}


void Application::Initialize(uint width, uint height)
{
	if (s_Initialized) return;

	// Set the window and render dimensions.
	s_RenderWidth = width, s_WindowWidth = width;
	s_RenderHeight = height, s_WindowHeight = height;

	Application::InitGLFW();
	Application::InitImGui();
	Application::InitOpenCL();
	Input::Initialize(Application::Window());

	s_RenderSurface = new Surface(s_RenderWidth, s_RenderHeight);

	// Set the Game to be initialized.
	s_Initialized = true;

}

void Application::Run()
{
	// Initialize with some default width and height if the app was not yet intialized.
	if (!s_Initialized) Initialize(1024, 1024);

	// Initialize the game. 
	Game* game = new Game();

	// Variables for computing time passed per frame.
	std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
	std::chrono::system_clock::time_point tc = std::chrono::system_clock::now();
	float dt = std::chrono::duration<float>(tc - tp).count() + 0.00001f;

	while (!Input::KeyPressed(Key::Escape) && !glfwWindowShouldClose(Application::Window())) {
		// Compute the time passed since last loop.
		float dt = std::chrono::duration<float>(tc - tp).count() + 0.00001f;
		tp = tc; tc = std::chrono::system_clock::now();
		
		glClearColor(0.102f, 0.117f, 0.141f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT);


		game->Tick(dt);
		game->Draw(dt);


		// Render our render-target.
		Application::Screen()->Draw();

		game->RenderGUI(dt);

		// Update the Input singleton.
		Input::Update();

		glfwSwapBuffers(Application::Window());
		glfwPollEvents();
	}

	delete game;
}

GLFWwindow* Application::Window()
{
	return s_Window;
}

Surface* Application::Screen()
{
	return s_RenderSurface;
}

clContext* Application::CLcontext()
{
	return s_clContext;
}

uint Application::WindowWidth()
{
	return s_WindowWidth;
}

uint Application::WindowHeight()
{
	return s_WindowHeight;
}

uint Application::RenderWidth()
{
	return s_RenderWidth;
}

uint Application::RenderHeight()
{
	return s_RenderHeight;
}

void Application::InitGLFW()
{
	// Initialize glfw.
	glewExperimental = true;

	if (!glfwInit()) FATAL_ERROR("Failed to initialize glfw.");

	// Set the window variables.
	glfwWindowHint(GLFW_SAMPLES, 4);				// 4x antialiasing
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL 
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	s_Window = glfwCreateWindow(s_WindowWidth, s_WindowHeight, "Sandbox", NULL, NULL);
	if (s_Window == NULL) FATAL_ERROR("Failed to create GLFW window.");

	glfwSetWindowAspectRatio(s_Window, s_WindowWidth, s_WindowHeight);

	// Initialize glew.
	glfwMakeContextCurrent(s_Window);

	if (glewInit() != GLEW_OK) FATAL_ERROR("Failed to initialize glew.");

	// Enable OpenGL debug callbacks.
	EnableGLdebugInfo();

	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

	// Set the input modes.
	glfwSetInputMode(s_Window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetInputMode(s_Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

	glfwSwapInterval(0);
}

void Application::InitImGui()
{
	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();

	// Setup Platform/Renderer bindings
	ImGui_ImplGlfw_InitForOpenGL(s_Window, true);
	ImGui_ImplOpenGL3_Init("#version 330");
	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
}

void Application::InitOpenCL()
{
	s_clContext = new clContext(true);
}
