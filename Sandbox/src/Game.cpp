#include "stdfax.h"
#include <glm/gtx/compatibility.hpp>
#include "Game.h"

#define DX	(1.0f / 32.0f)
#define RDX (1.0f / DX)
#define HALFDX (0.5f * DX)
#define VISCOSITY 1.0f
#define TIMESTEP 0.05f		// 20 simulation steps per "unit" time-measure at least.

#define EPSILON 1e-4f
#define NUM_THREADS 12

Game::Game()
{
	m_VelocityBuffer = (glm::vec2*)malloc(sizeof(glm::vec2) * WIDTH * HEIGHT);
	m_VelocityOutput = (glm::vec2*)malloc(sizeof(glm::vec2) * WIDTH * HEIGHT);
	m_PressureBuffer = (float*)malloc(sizeof(float) * WIDTH * HEIGHT);
	m_PressureOutput = (float*)malloc(sizeof(float) * WIDTH * HEIGHT);
	m_ColorBuffer = (glm::vec4*)malloc(sizeof(glm::vec4) * WIDTH * HEIGHT);
	m_ColorOutput = (glm::vec4*)malloc(sizeof(glm::vec4) * WIDTH * HEIGHT);
	m_DivergenceBuffer = (float*)malloc(sizeof(float) * WIDTH * HEIGHT);

	InitSimulation();
}

Game::~Game()
{
	free(m_VelocityBuffer);
	free(m_VelocityOutput);
	free(m_PressureBuffer);
	free(m_PressureOutput);
	free(m_ColorBuffer);
	free(m_ColorOutput);
	free(m_DivergenceBuffer);
}

void Game::Tick(float dt)
{
	HandleInput(dt);

	// Clamp the timestep to a maximum of TIMESTEP.
	dt = glm::min(dt, TIMESTEP);
	SimulateTimeStep(dt);

}

void Game::Draw(float dt)
{
	Application::Screen()->PlotPixels((Color*)m_ColorBuffer);
	Application::Screen()->SyncPixels();
}

void Game::RenderGUI(float dt)
{
	// GUI code goes here. 

	// feed inputs to dear imgui, start new frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	const static char* windowTitle = "Debug";
	static bool display = true;
	ImGui::Begin(windowTitle, &display, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
	ImGui::SetWindowFontScale(1.75f);
	ImGui::Text("Frame-time: %.1f", dt * 1000.0f);
	ImGui::End();

	// Render dear imgui into screen
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Game::InitSimulation()
{
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {

			m_PressureBuffer[x + y * WIDTH] = 0.0f;
			m_VelocityBuffer[x + y * WIDTH] = glm::vec2(0.0f, 0.0f);
			m_ColorBuffer[x + y * WIDTH] = glm::vec4(0.0f);
		}
	}
}

void Game::SimulateTimeStep(float dt)
{
	// Update the velocities.
	UpdateVelocityBoundaries();
	AdvectVelocity(dt);
	memcpy(m_VelocityBuffer, m_VelocityOutput, sizeof(glm::vec2) * WIDTH * HEIGHT);

	for (int i = 0; i < 8; i++) {
		DiffuseVelocities(dt);
		memcpy(m_VelocityBuffer, m_VelocityOutput, sizeof(glm::vec2) * WIDTH * HEIGHT);
	}

	// Update divergence.
	ComputeDivergence();

	for (int i = 0; i < 8; i++) {
		ComputePressure();
		memcpy(m_PressureBuffer, m_PressureOutput, sizeof(float) * WIDTH * HEIGHT);
	}
	UpdatePressureBoundaries();

	SubtractPressureGradient();
	ResetPressureGrid();

	UpdateColorBoundaries();
	AdvectColors(dt);
	memcpy(m_ColorBuffer, m_ColorOutput, sizeof(glm::vec4) * WIDTH * HEIGHT);
}

void Game::HandleInput(float dt)
{
	HandleMouseDown(dt);
	HandleMouseClick(dt);

	if (Input::KeyPressed(Key::R)) InitSimulation();
}

void Game::HandleMouseDown(float dt)
{
	static const float sqrdRad = RDX * RDX * 0.75f;

	// Only apply forces when the mouse is being pressed.
	if (!Input::MouseLeftButtonDown()) return;

	// Check if mouse is inside the screen. 
	glm::ivec2 cursorPos = Input::CursorPosition();
	cursorPos.y = HEIGHT - cursorPos.y - 1.0f;

	if (cursorPos.x < 0 || cursorPos.y < 0 || cursorPos.x > WIDTH - 1 || cursorPos.y > HEIGHT - 1) return;

	// Force-direction.
	glm::vec2 forceDirection = Input::CursorMovement();
	forceDirection.y *= -1.0f;

	if (glm::abs(forceDirection.x) < EPSILON || glm::abs(forceDirection.y) < EPSILON) return;
	forceDirection = glm::normalize(forceDirection);

	glm::ivec2 minBounds = glm::clamp(cursorPos - 100, glm::ivec2(0), glm::ivec2(WIDTH - 1, HEIGHT - 1));
	glm::ivec2 maxBounds = glm::clamp(cursorPos + 100, glm::ivec2(0), glm::ivec2(WIDTH - 1, HEIGHT - 1));

	for (int dy = minBounds.y; dy < maxBounds.y; dy++)
		for (int dx = minBounds.x; dx < maxBounds.x; dx++) {
			float sqrdDist =
				(dx - cursorPos.x) * (dx - cursorPos.x) +
				(dy - cursorPos.y) * (dy - cursorPos.y);

			float multiplier = 1.0f;

			if (sqrdDist > sqrdRad) continue;
			if (sqrdDist > (0.5f * sqrdRad)) multiplier = 5.0f;
			if (sqrdDist > (0.2f * sqrdRad)) multiplier = 10.0f;

			// Update the velocity and color.
			m_VelocityBuffer[dx + dy * WIDTH] = forceDirection * multiplier;
			m_ColorBuffer[dx + dy * WIDTH] = glm::vec4(1.0f);
		}
}

void Game::HandleMouseClick(float dt)
{
	static const float maxRad = RDX * RDX;
	static const float minRad = maxRad * 0.5f;


	// Only apply forces when the mouse is being pressed.
	if (!Input::MouseRightButtonClick()) return;

	// Check if mouse is inside the screen. 
	glm::ivec2 cursorPos = Input::CursorPosition();
	cursorPos.y = HEIGHT - cursorPos.y - 1.0f;

	if (cursorPos.x < 0 || cursorPos.y < 0 || cursorPos.x > WIDTH - 1 || cursorPos.y > HEIGHT - 1) return;

	glm::ivec2 minBounds = glm::clamp(cursorPos - 100, glm::ivec2(0), glm::ivec2(WIDTH - 1, HEIGHT - 1));
	glm::ivec2 maxBounds = glm::clamp(cursorPos + 100, glm::ivec2(0), glm::ivec2(WIDTH - 1, HEIGHT - 1));

	for (int dy = minBounds.y; dy < maxBounds.y; dy++)
		for (int dx = minBounds.x; dx < maxBounds.x; dx++) {
			float sqrdDist =
				(dx - cursorPos.x) * (dx - cursorPos.x) +
				(dy - cursorPos.y) * (dy - cursorPos.y);

			if (dx == cursorPos.x && dy == cursorPos.y) continue;

			glm::vec2 force = glm::normalize(glm::vec2((float)dx - cursorPos.x, (float)dy - cursorPos.y)) * 10.0f;

			// Update the velocity and color.
			m_VelocityBuffer[dx + dy * WIDTH] = force;
			if (sqrdDist < minRad || sqrdDist > maxRad) continue;
			m_ColorBuffer[dx + dy * WIDTH] = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
		}
}


void Game::UpdateVelocityBoundaries()
{
	const float scale = -1.0f;
	// Loop over the x-boundaries.
	for (int x = 0; x < WIDTH; x++) {
		// Update the boundaries. 
		m_VelocityBuffer[x + 0 * WIDTH] = m_VelocityBuffer[x + 1 * WIDTH] * scale;
		m_VelocityBuffer[x + (HEIGHT - 1) * WIDTH] = m_VelocityBuffer[x + (HEIGHT - 2) * WIDTH] * scale;
	}
	// Loop over the y-boundaries.
	for (int y = 0; y < HEIGHT; y++) {
		// Update the boundaries.
		m_VelocityBuffer[0 + y * WIDTH] = m_VelocityBuffer[1 + y * WIDTH] * scale;
		m_VelocityBuffer[(WIDTH - 1) + y * WIDTH] = m_VelocityBuffer[(WIDTH - 2) + y * WIDTH] * scale;
	}
}

void Game::AdvectVelocity(float dt)
{
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {

			const float fWidth = (float)WIDTH;
			const float fHeight = (float)HEIGHT;

			glm::vec2 pos = glm::vec2(x, y) - dt * RDX * m_VelocityBuffer[x + y * WIDTH];

			int stx = (int)glm::clamp(floor(pos.x), 0.0f, fWidth - 1.0f);
			int sty = (int)glm::clamp(floor(pos.y), 0.0f, fHeight - 1.0f);
			int stz = (int)glm::clamp(stx + 1.0f, 0.0f, fWidth - 1.0f);
			int stw = (int)glm::clamp(sty + 1.0f, 0.0f, fHeight - 1.0f);

			glm::vec2 t = glm::vec2(glm::clamp(pos.x - stx, 0.0f, 1.0f), glm::clamp(pos.y - sty, 0.0f, 1.0f));

			glm::vec2 v1 = m_VelocityBuffer[stx + sty * WIDTH];
			glm::vec2 v2 = m_VelocityBuffer[stz + sty * WIDTH];
			glm::vec2 v3 = m_VelocityBuffer[stx + stw * WIDTH];
			glm::vec2 v4 = m_VelocityBuffer[stz + stw * WIDTH];

			m_VelocityOutput[x + y * WIDTH] = glm::lerp(glm::lerp(v1, v2, t.x), glm::lerp(v3, v4, t.x), t.y);
		}
	}
}

void Game::DiffuseVelocities(float dt)
{
	float alpha = (DX * DX) / (VISCOSITY * dt);
	float rBeta = 1.0f / (alpha + 4.0f);

#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {

			int stx = glm::clamp(x - 1, 0, WIDTH - 1);
			int sty = glm::clamp(y - 1, 0, HEIGHT - 1);
			int stz = glm::clamp(x + 1, 0, WIDTH - 1);
			int stw = glm::clamp(y + 1, 0, HEIGHT - 1);

			// Retrieve the four samples.
			glm::vec2 xL = m_VelocityBuffer[stx + y * WIDTH];
			glm::vec2 xR = m_VelocityBuffer[stz + y * WIDTH];
			glm::vec2 xB = m_VelocityBuffer[x + sty * WIDTH];
			glm::vec2 xT = m_VelocityBuffer[x + stw * WIDTH];

			// Sample b from the center.
			glm::vec2 bC = m_VelocityBuffer[x + y * WIDTH];

			// Evaluate the Jacobi iteration. 
			m_VelocityOutput[x + y * WIDTH] = (xL + xR + xB + xT + alpha * bC) * rBeta;
		}
	}
}

void Game::ComputeDivergence()
{
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			int stx = glm::clamp(x - 1, 0, WIDTH - 1);
			int sty = glm::clamp(y - 1, 0, HEIGHT - 1);
			int stz = glm::clamp(x + 1, 0, WIDTH - 1);
			int stw = glm::clamp(y + 1, 0, HEIGHT - 1);

			glm::vec2 wL = m_VelocityBuffer[stx + y * WIDTH];
			glm::vec2 wR = m_VelocityBuffer[stz + y * WIDTH];
			glm::vec2 wB = m_VelocityBuffer[x + sty * WIDTH];
			glm::vec2 wT = m_VelocityBuffer[x + stw * WIDTH];

			m_DivergenceBuffer[x + y * WIDTH] = HALFDX * ((wR.x - wL.x) + (wT.y - wB.y));
		}
	}
}

void Game::ComputePressure()
{
	float alpha = -1.0f * (DX * DX);
	float rBeta = 0.25f;

#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			int stx = glm::clamp(x - 1, 0, WIDTH - 1);
			int sty = glm::clamp(y - 1, 0, HEIGHT - 1);
			int stz = glm::clamp(x + 1, 0, WIDTH - 1);
			int stw = glm::clamp(y + 1, 0, HEIGHT - 1);

			// Retrieve the four samples.
			float xL = m_PressureBuffer[stx + y * WIDTH];
			float xR = m_PressureBuffer[stz + y * WIDTH];
			float xB = m_PressureBuffer[x + sty * WIDTH];
			float xT = m_PressureBuffer[x + stw * WIDTH];

			// Sample b from the center.
			float bC = m_DivergenceBuffer[x + y * WIDTH];

			// Evaluate the Jacobi iteration. 
			m_PressureOutput[x + y * WIDTH] = (xL + xR + xB + xT + alpha * bC) * rBeta;
		}
	}

}

void Game::UpdatePressureBoundaries()
{
	const float scale = 1.0f;
	// Loop over the x-boundaries.
	for (int x = 0; x < WIDTH; x++) {
		// Update the boundaries. 
		m_PressureBuffer[x + 0 * WIDTH] = m_PressureBuffer[x + 1 * WIDTH] * scale;
		m_PressureBuffer[x + (HEIGHT - 1) * WIDTH] = m_PressureBuffer[x + (HEIGHT - 2) * WIDTH] * scale;
	}
	// Loop over the y-boundaries.
	for (int y = 0; y < HEIGHT; y++) {
		// Update the boundaries.
		m_PressureBuffer[0 + y * WIDTH] = m_PressureBuffer[1 + y * WIDTH] * scale;
		m_PressureBuffer[(WIDTH - 1) + y * WIDTH] = m_PressureBuffer[(WIDTH - 2) + y * WIDTH] * scale;
	}
}

void Game::SubtractPressureGradient()
{
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {

			int stx = glm::clamp(x - 1, 0, WIDTH - 1);
			int sty = glm::clamp(y - 1, 0, HEIGHT - 1);
			int stz = glm::clamp(x + 1, 0, WIDTH - 1);
			int stw = glm::clamp(y + 1, 0, HEIGHT - 1);

			float pL = m_PressureBuffer[stx + y * WIDTH];
			float pR = m_PressureBuffer[stz + y * WIDTH];
			float pB = m_PressureBuffer[x + sty * WIDTH];
			float pT = m_PressureBuffer[x + stw * WIDTH];

			m_VelocityBuffer[x + y * WIDTH] = m_VelocityBuffer[x + y * WIDTH] - HALFDX * glm::vec2(pR - pL, pT - pB);
		}
	}
}

void Game::ResetPressureGrid()
{
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			m_PressureBuffer[x + y * WIDTH] = 0.0f;
		}
	}
}

void Game::UpdateColorBoundaries()
{
	const float scale = 0.0f;
	// Loop over the x-boundaries.
	for (int x = 0; x < WIDTH; x++) {
		// Update the boundaries. 
		m_ColorBuffer[x + 0 * WIDTH] = m_ColorBuffer[x + 1 * WIDTH] * scale;
		m_ColorBuffer[x + (HEIGHT - 1) * WIDTH] = m_ColorBuffer[x + (HEIGHT - 2) * WIDTH] * scale;
	}
	// Loop over the y-boundaries.
	for (int y = 0; y < HEIGHT; y++) {
		// Update the boundaries.
		m_ColorBuffer[0 + y * WIDTH] = m_ColorBuffer[1 + y * WIDTH] * scale;
		m_ColorBuffer[(WIDTH - 1) + y * WIDTH] = m_ColorBuffer[(WIDTH - 2) + y * WIDTH] * scale;
	}
}

void Game::AdvectColors(float dt)
{
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {

			static const float fWidth = (float)WIDTH;
			static const float fHeight = (float)HEIGHT;

			glm::vec2 pos = glm::vec2(x, y) - dt * RDX * m_VelocityBuffer[x + y * WIDTH];

			int stx = (int)glm::clamp(floor(pos.x), 0.0f, fWidth - 1.0f);
			int sty = (int)glm::clamp(floor(pos.y), 0.0f, fHeight - 1.0f);
			int stz = (int)glm::clamp(stx + 1.0f, 0.0f, fWidth - 1.0f);
			int stw = (int)glm::clamp(sty + 1.0f, 0.0f, fHeight - 1.0f);

			glm::vec2 t = glm::vec2(glm::clamp(pos.x - stx, 0.0f, 1.0f), glm::clamp(pos.y - sty, 0.0f, 1.0f));

			glm::vec4 v1 = m_ColorBuffer[stx + sty * WIDTH];
			glm::vec4 v2 = m_ColorBuffer[stz + sty * WIDTH];
			glm::vec4 v3 = m_ColorBuffer[stx + stw * WIDTH];
			glm::vec4 v4 = m_ColorBuffer[stz + stw * WIDTH];

			m_ColorOutput[x + y * WIDTH] = glm::lerp(glm::lerp(v1, v2, t.x), glm::lerp(v3, v4, t.x), t.y);
		}
	}
}