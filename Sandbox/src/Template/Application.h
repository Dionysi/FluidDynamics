#pragma once
#include "Surface.h"
#include "Input.h"

#define WIDTH 1024
#define HEIGHT 1024

class Application
{
public:
	/*
	* Initialize the Game singleton.
	* @param[in] width			Window and render width.
	* @param[in] height			Window and render height.
	*/
	static void Initialize(uint width, uint height);
	/*
	* Start the application main-loop.
	*/
	static void Run();

	/*
	* Retrieve the active GLFW window.
	*/
	static GLFWwindow* Window();
	/*
	* Retrieve the active render-surface.
	*/
	static Surface* Screen();
	/*
	* Retrieve the global cl context.
	* @returns		Valid cl context object.
	*/
	static clContext* CLcontext();

	/*
	* Retrieve the window's width.
	* @returns		Window width in pixels.
	*/
	static uint WindowWidth();
	/*
	* Retrieve the window's height.
	* @returns		Window height in pixels.
	*/
	static uint WindowHeight();
	/*
	* Retrieve the window's render width.
	* @returns		Window render width in pixels.
	*/
	static uint RenderWidth();
	/*
	* Retrieve the window's render height.
	* @returns		Window render height in pixels.
	*/
	static uint RenderHeight();

private:
	/*
	* Global OpenCL context.
	*/
	static clContext* s_clContext;

	/*
	* Window size.
	*/
	static uint s_WindowWidth, s_WindowHeight;
	/*
	* Render width and height.
	*/
	static uint s_RenderWidth, s_RenderHeight;

	/*
	* Boolean indicating if the Game class has been intialized yet.
	*/
	static bool s_Initialized;

	/*
	* Active window.
	*/
	static GLFWwindow* s_Window;
	/*
	* Surface used for drawing to the window.
	*/
	static Surface* s_RenderSurface;

	/*
	* Initialize OpenGL and GLFW.
	*/
	static void InitGLFW();
	/*
	* Initializes the Dear ImGui library.
	*/
	static void InitImGui();
	/*
	* Initializes the OpenCL context.
	*/
	static void InitOpenCL();
};