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
	* Buffer containing the velocity values per grid cell.
	*/
	glm::vec2* m_VelocityBuffer = nullptr, * m_VelocityOutput = nullptr;
	/*
	* Buffer containing the pressure values per grid cell.
	*/
	float* m_PressureBuffer = nullptr, * m_PressureOutput = nullptr;
	/*
	* Buffer containing the color values per grid cell.
	*/
	glm::vec4* m_ColorBuffer = nullptr, * m_ColorOutput = nullptr;
	/*
	* Buffer storing the divergence values.
	*/
	float* m_DivergenceBuffer = nullptr;

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

	void UpdateVelocityBoundaries();
	void AdvectVelocity(float dt);
	void DiffuseVelocities(float dt);
	void ComputeDivergence();
	void ComputePressure();
	void UpdatePressureBoundaries();
	void SubtractPressureGradient();
	void UpdateColorBoundaries();
	void AdvectColors(float dt);
};

