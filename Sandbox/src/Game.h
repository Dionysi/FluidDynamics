#pragma once
#include "Template/Application.h"

class Game
{

public:
	Game();
	~Game();
	void Tick(float dt);
	void Draw(float dt);
	void RenderGUI(float dt);

private:
	/*
	* OpenCL command queue for running our OpenCL kernels.
	*/
	clCommandQueue* m_CommandQueue;
	/*
	* Our OpenCL program which contains all OpenCL code.
	* The program is used to create OpenCL kernels.
	*/
	clProgram* m_Program;

	// Create a buffer for each of the input-output buffers.
	clBuffer* m_VelocityInputBuffer, * m_VelocityOutputBuffer;
	clBuffer* m_PressureInputBuffer, * m_PressureOutputBuffer;
	clBuffer* m_ColorInputBuffer, * m_ColorOutputBuffer;
	clBuffer* m_DivergenceInputBuffer;
	
	// Rendering
	clBuffer* m_RenderBuffer;
	clKernel* m_CopyKernel;

	// Create a kernel for each simulation step.
	clKernel* m_UpdateVelocityBoundariesKernel;
	clKernel* m_AdvectVelocityKernel;
	clKernel* m_DiffuseVelocitiesKernel;
	clKernel* m_ComputeDivergenceKernel;
	clKernel* m_ComputePressureKernel;
	clKernel* m_UpdatePressureBoundariesKernel;
	clKernel* m_SubtractPressureGradientKernel;
	clKernel* m_UpdateColorBoundariesKernel;
	clKernel* m_AdvectColorsKernel;

	clKernel* m_HandleMouseDownKernel;
	clKernel* m_HandleMouseClickKernel;

	/*
	* Buffer containing the velocity values per grid cell.
	*/
	glm::vec2* m_VelocityInput = nullptr, * m_VelocityOutput = nullptr;
	/*
	* Buffer containing the pressure values per grid cell.
	*/
	float* m_PressureInput = nullptr, * m_PressureOutput = nullptr;
	/*
	* Buffer containing the color values per grid cell.
	*/
	glm::vec4* m_ColorInput = nullptr, * m_ColorOutput = nullptr;
	/*
	* Buffer storing the divergence values.
	*/
	float* m_DivergenceInput = nullptr;

	/*
	* Initialize simulation values.
	*/
	void InitSimulation();
	/*
	* Simulate a time-step.
	*/
	void SimulateTimeStep(float dt);
	/* 
	* Apply forces based on the user-input.
	*/
	void HandleInput(float dt);
	/*
	* Apply forces when the mouse was held-down.
	*/
	void HandleMouseDown(float dt);
	/*
	* Apply forces when the mouse left-button was clicked.
	*/
	void HandleMouseClick(float dt);
};

