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

int work_dim = 2;
size_t global_size[2] = { WIDTH , HEIGHT };
size_t local_size[2] = { 32 , 32 };

Game::Game()
{
	// Initialize our OpenCL context.
	m_CommandQueue = new clCommandQueue(Application::CLcontext(), false, false);
	m_Program = new clProgram(Application::CLcontext(), "kernels.cl");

	Application::CLcontext()->PrintDeviceInfo();

	// Initialize our buffers.
	m_VelocityInput = (glm::vec2*)malloc(sizeof(glm::vec2) * WIDTH * HEIGHT);
	m_VelocityInputBuffer = new clBuffer(Application::CLcontext(), sizeof(glm::vec2) * WIDTH * HEIGHT, BufferFlags::READ_WRITE);

	m_VelocityOutput = (glm::vec2*)malloc(sizeof(glm::vec2) * WIDTH * HEIGHT);
	m_VelocityOutputBuffer = new clBuffer(Application::CLcontext(), sizeof(glm::vec2) * WIDTH * HEIGHT, BufferFlags::READ_WRITE);

	m_PressureInput = (float*)malloc(sizeof(float) * WIDTH * HEIGHT);
	m_PressureInputBuffer = new clBuffer(Application::CLcontext(), sizeof(float) * WIDTH * HEIGHT, BufferFlags::READ_WRITE);

	m_PressureOutput = (float*)malloc(sizeof(float) * WIDTH * HEIGHT);
	m_PressureOutputBuffer = new clBuffer(Application::CLcontext(), sizeof(float) * WIDTH * HEIGHT, BufferFlags::READ_WRITE);

	m_ColorInput = (glm::vec4*)malloc(sizeof(glm::vec4) * WIDTH * HEIGHT);
	m_ColorInputBuffer = new clBuffer(Application::CLcontext(), sizeof(glm::vec4) * WIDTH * HEIGHT, BufferFlags::READ_WRITE);

	m_ColorOutput = (glm::vec4*)malloc(sizeof(glm::vec4) * WIDTH * HEIGHT);
	m_ColorOutputBuffer = new clBuffer(Application::CLcontext(), sizeof(glm::vec4) * WIDTH * HEIGHT, BufferFlags::READ_WRITE);

	m_DivergenceInput = (float*)malloc(sizeof(float) * WIDTH * HEIGHT);
	m_DivergenceInputBuffer = new clBuffer(Application::CLcontext(), sizeof(float) * WIDTH * HEIGHT, BufferFlags::READ_WRITE);

	// Create our kernels and set arguments per-kernel.
	m_UpdateVelocityBoundariesKernel = new clKernel(m_Program, "UpdateVelocityBoundaries");
	m_UpdateVelocityBoundariesKernel->SetArgument(0, m_VelocityInputBuffer);

	m_AdvectVelocityKernel = new clKernel(m_Program, "AdvectVelocity");
	m_AdvectVelocityKernel->SetArgument(1, m_VelocityInputBuffer);
	m_AdvectVelocityKernel->SetArgument(2, m_VelocityOutputBuffer);

	m_DiffuseVelocitiesKernel = new clKernel(m_Program, "DiffuseVelocities");
	m_DiffuseVelocitiesKernel->SetArgument(1, m_VelocityInputBuffer);
	m_DiffuseVelocitiesKernel->SetArgument(2, m_VelocityOutputBuffer);

	m_ComputeDivergenceKernel = new clKernel(m_Program, "ComputeDivergence");
	m_ComputeDivergenceKernel->SetArgument(0, m_VelocityInputBuffer);
	m_ComputeDivergenceKernel->SetArgument(1, m_DivergenceInputBuffer);

	m_ComputePressureKernel = new clKernel(m_Program, "ComputePressure");
	m_ComputePressureKernel->SetArgument(0, m_PressureInputBuffer);
	m_ComputePressureKernel->SetArgument(1, m_PressureOutputBuffer);
	m_ComputePressureKernel->SetArgument(2, m_DivergenceInputBuffer);

	m_UpdatePressureBoundariesKernel = new clKernel(m_Program, "UpdatePressureBoundaries");
	m_UpdatePressureBoundariesKernel->SetArgument(0, m_PressureInputBuffer);

	m_SubtractPressureGradientKernel = new clKernel(m_Program, "SubtractPressureGradient");
	m_SubtractPressureGradientKernel->SetArgument(0, m_PressureInputBuffer);
	m_SubtractPressureGradientKernel->SetArgument(1, m_VelocityInputBuffer);

	m_UpdateColorBoundariesKernel = new clKernel(m_Program, "UpdateColorBoundaries");
	m_UpdateColorBoundariesKernel->SetArgument(0, m_ColorInputBuffer);

	m_AdvectColorsKernel = new clKernel(m_Program, "AdvectColors");
	m_AdvectColorsKernel->SetArgument(1, m_ColorInputBuffer);
	m_AdvectColorsKernel->SetArgument(2, m_ColorOutputBuffer);
	m_AdvectColorsKernel->SetArgument(3, m_VelocityInputBuffer);


	InitSimulation();
}

Game::~Game()
{
	free(m_VelocityInput);
	free(m_VelocityOutput);
	free(m_PressureInput);
	free(m_PressureOutput);
	free(m_ColorInput);
	free(m_ColorOutput);
	free(m_DivergenceInput);

	delete m_VelocityInputBuffer;
	delete m_VelocityOutputBuffer;
	delete m_PressureInputBuffer;
	delete m_PressureOutputBuffer;
	delete m_ColorInputBuffer;
	delete m_ColorOutputBuffer;
	delete m_DivergenceInputBuffer;

	delete m_Program;
	delete m_CommandQueue;
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
	Application::Screen()->PlotPixels((Color*)m_ColorInput);
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

			m_PressureInput[x + y * WIDTH] = 0.0f;
			m_VelocityInput[x + y * WIDTH] = glm::vec2(0.0f, 0.0f);
			m_ColorInput[x + y * WIDTH] = glm::vec4(0.0f);
		}
	}
}

void Game::SimulateTimeStep(float dt)
{
	// Copy kernel data.
	m_VelocityInputBuffer->CopyToDevice(m_CommandQueue, m_VelocityInput, false);
	m_VelocityOutputBuffer->CopyToDevice(m_CommandQueue, m_VelocityOutput, false);
	m_PressureInputBuffer->CopyToDevice(m_CommandQueue, m_PressureInput, false);
	m_PressureOutputBuffer->CopyToDevice(m_CommandQueue, m_PressureOutput, false);
	m_ColorInputBuffer->CopyToDevice(m_CommandQueue, m_ColorInput, false);
	m_ColorOutputBuffer->CopyToDevice(m_CommandQueue, m_ColorOutput, false);
	m_DivergenceInputBuffer->CopyToDevice(m_CommandQueue, m_DivergenceInput, true);


	//UpdateVelocityBoundaries();
	m_UpdateVelocityBoundariesKernel->Enqueue(m_CommandQueue, glm::max(WIDTH, HEIGHT), 1024); 

	//AdvectVelocity(dt);
	m_AdvectVelocityKernel->SetArgument(0, &dt, sizeof(float));
	m_AdvectVelocityKernel->Enqueue(m_CommandQueue, work_dim, global_size, local_size);

	//memcpy(m_VelocityInput, m_VelocityOutput, sizeof(glm::vec2) * WIDTH * HEIGHT);
	clBuffer::CopyBufferToBuffer(m_CommandQueue, m_VelocityInputBuffer, m_VelocityOutputBuffer, m_VelocityInputBuffer->GetSize());

	// First set the param for the diffuse-velocities kernel once.
	m_DiffuseVelocitiesKernel->SetArgument(0, &dt, sizeof(float));

	for (int i = 0; i < 8; i++) {
		//DiffuseVelocities(dt);
		m_DiffuseVelocitiesKernel->Enqueue(m_CommandQueue, work_dim, global_size, local_size);

		//memcpy(m_VelocityInput, m_VelocityOutput, sizeof(glm::vec2) * WIDTH * HEIGHT);
		clBuffer::CopyBufferToBuffer(m_CommandQueue, m_VelocityInputBuffer, m_VelocityOutputBuffer, m_VelocityInputBuffer->GetSize());
	}


	//ComputeDivergence();
	m_ComputeDivergenceKernel->Enqueue(m_CommandQueue, work_dim, global_size, local_size);

	for (int i = 0; i < 8; i++) {
		//ComputePressure();
		m_ComputePressureKernel->Enqueue(m_CommandQueue, work_dim, global_size, local_size);

		//memcpy(m_PressureInput, m_PressureOutput, sizeof(float) * WIDTH * HEIGHT);
		clBuffer::CopyBufferToBuffer(m_CommandQueue, m_PressureInputBuffer, m_PressureOutputBuffer, m_PressureInputBuffer->GetSize());
	}

	//UpdatePressureBoundaries();
	m_UpdatePressureBoundariesKernel->Enqueue(m_CommandQueue, glm::max(WIDTH, HEIGHT), 1024);

	//SubtractPressureGradient();
	m_SubtractPressureGradientKernel->Enqueue(m_CommandQueue, work_dim, global_size, local_size);

	//UpdateColorBoundaries();
	m_UpdateColorBoundariesKernel->Enqueue(m_CommandQueue, glm::max(WIDTH, HEIGHT), 1024);


	//AdvectColors(dt);
	m_AdvectColorsKernel->SetArgument(0, &dt, sizeof(float));
	m_AdvectColorsKernel->Enqueue(m_CommandQueue, work_dim, global_size, local_size);	
	//memcpy(m_ColorInput, m_ColorOutput, sizeof(glm::vec4) * WIDTH * HEIGHT);
	clBuffer::CopyBufferToBuffer(m_CommandQueue, m_ColorInputBuffer, m_ColorOutputBuffer, m_ColorInputBuffer->GetSize());

	m_VelocityInputBuffer->CopyToHost(m_CommandQueue, m_VelocityInput, false);
	m_VelocityOutputBuffer->CopyToHost(m_CommandQueue, m_VelocityOutput, false);
	m_PressureInputBuffer->CopyToHost(m_CommandQueue, m_PressureInput, false);
	m_PressureOutputBuffer->CopyToHost(m_CommandQueue, m_PressureOutput, false);
	m_ColorInputBuffer->CopyToHost(m_CommandQueue, m_ColorInput, false);
	m_ColorOutputBuffer->CopyToHost(m_CommandQueue, m_ColorOutput, false);
	m_DivergenceInputBuffer->CopyToHost(m_CommandQueue, m_DivergenceInput, true);
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
			m_VelocityInput[dx + dy * WIDTH] = forceDirection * multiplier;
			m_ColorInput[dx + dy * WIDTH] = glm::vec4(1.0f);
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
			m_VelocityInput[dx + dy * WIDTH] = force;
			if (sqrdDist < minRad || sqrdDist > maxRad) continue;
			m_ColorInput[dx + dy * WIDTH] = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
		}
}


void Game::UpdateVelocityBoundaries()
{

	const float scale = -1.0f;
	// Loop over the x-boundaries.
	for (int x = 0; x < WIDTH; x++) {
		// Update the boundaries. 
		m_VelocityInput[x + 0 * WIDTH] = m_VelocityInput[x + 1 * WIDTH] * scale;
		m_VelocityInput[x + (HEIGHT - 1) * WIDTH] = m_VelocityInput[x + (HEIGHT - 2) * WIDTH] * scale;
	}
	// Loop over the y-boundaries.
	for (int y = 0; y < HEIGHT; y++) {
		// Update the boundaries.
		m_VelocityInput[0 + y * WIDTH] = m_VelocityInput[1 + y * WIDTH] * scale;
		m_VelocityInput[(WIDTH - 1) + y * WIDTH] = m_VelocityInput[(WIDTH - 2) + y * WIDTH] * scale;
	}
}

void Game::AdvectVelocity(float dt)
{

	#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
		for (int y = 0; y < HEIGHT; y++) {
			for (int x = 0; x < WIDTH; x++) {
	
				const float fWidth = (float)WIDTH;
				const float fHeight = (float)HEIGHT;
	
				glm::vec2 pos = glm::vec2(x, y) - dt * RDX * m_VelocityInput[x + y * WIDTH];
	
				int stx = (int)glm::clamp(floor(pos.x), 0.0f, fWidth - 1.0f);
				int sty = (int)glm::clamp(floor(pos.y), 0.0f, fHeight - 1.0f);
				int stz = (int)glm::clamp(stx + 1.0f, 0.0f, fWidth - 1.0f);
				int stw = (int)glm::clamp(sty + 1.0f, 0.0f, fHeight - 1.0f);
	
				glm::vec2 t = glm::vec2(glm::clamp(pos.x - stx, 0.0f, 1.0f), glm::clamp(pos.y - sty, 0.0f, 1.0f));
	
				glm::vec2 v1 = m_VelocityInput[stx + sty * WIDTH];
				glm::vec2 v2 = m_VelocityInput[stz + sty * WIDTH];
				glm::vec2 v3 = m_VelocityInput[stx + stw * WIDTH];
				glm::vec2 v4 = m_VelocityInput[stz + stw * WIDTH];
	
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
				glm::vec2 xL = m_VelocityInput[stx + y * WIDTH];
				glm::vec2 xR = m_VelocityInput[stz + y * WIDTH];
				glm::vec2 xB = m_VelocityInput[x + sty * WIDTH];
				glm::vec2 xT = m_VelocityInput[x + stw * WIDTH];
	
				// Sample b from the center.
				glm::vec2 bC = m_VelocityInput[x + y * WIDTH];
	
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

			glm::vec2 wL = m_VelocityInput[stx + y * WIDTH];
			glm::vec2 wR = m_VelocityInput[stz + y * WIDTH];
			glm::vec2 wB = m_VelocityInput[x + sty * WIDTH];
			glm::vec2 wT = m_VelocityInput[x + stw * WIDTH];

			m_DivergenceInput[x + y * WIDTH] = HALFDX * ((wR.x - wL.x) + (wT.y - wB.y));
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
			float xL = m_PressureInput[stx + y * WIDTH];
			float xR = m_PressureInput[stz + y * WIDTH];
			float xB = m_PressureInput[x + sty * WIDTH];
			float xT = m_PressureInput[x + stw * WIDTH];

			// Sample b from the center.
			float bC = m_DivergenceInput[x + y * WIDTH];

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
		m_PressureInput[x + 0 * WIDTH] = m_PressureInput[x + 1 * WIDTH] * scale;
		m_PressureInput[x + (HEIGHT - 1) * WIDTH] = m_PressureInput[x + (HEIGHT - 2) * WIDTH] * scale;
	}
	// Loop over the y-boundaries.
	for (int y = 0; y < HEIGHT; y++) {
		// Update the boundaries.
		m_PressureInput[0 + y * WIDTH] = m_PressureInput[1 + y * WIDTH] * scale;
		m_PressureInput[(WIDTH - 1) + y * WIDTH] = m_PressureInput[(WIDTH - 2) + y * WIDTH] * scale;
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

			float pL = m_PressureInput[stx + y * WIDTH];
			float pR = m_PressureInput[stz + y * WIDTH];
			float pB = m_PressureInput[x + sty * WIDTH];
			float pT = m_PressureInput[x + stw * WIDTH];

			m_VelocityInput[x + y * WIDTH] = m_VelocityInput[x + y * WIDTH] - HALFDX * glm::vec2(pR - pL, pT - pB);
		}
	}
}

void Game::UpdateColorBoundaries()
{
	const float scale = 0.0f;
	// Loop over the x-boundaries.
	for (int x = 0; x < WIDTH; x++) {
		// Update the boundaries. 
		m_ColorInput[x + 0 * WIDTH] = m_ColorInput[x + 1 * WIDTH] * scale;
		m_ColorInput[x + (HEIGHT - 1) * WIDTH] = m_ColorInput[x + (HEIGHT - 2) * WIDTH] * scale;
	}
	// Loop over the y-boundaries.
	for (int y = 0; y < HEIGHT; y++) {
		// Update the boundaries.
		m_ColorInput[0 + y * WIDTH] = m_ColorInput[1 + y * WIDTH] * scale;
		m_ColorInput[(WIDTH - 1) + y * WIDTH] = m_ColorInput[(WIDTH - 2) + y * WIDTH] * scale;
	}
}

void Game::AdvectColors(float dt)
{
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {

			static const float fWidth = (float)WIDTH;
			static const float fHeight = (float)HEIGHT;

			glm::vec2 pos = glm::vec2(x, y) - dt * RDX * m_VelocityInput[x + y * WIDTH];

			int stx = (int)glm::clamp(floor(pos.x), 0.0f, fWidth - 1.0f);
			int sty = (int)glm::clamp(floor(pos.y), 0.0f, fHeight - 1.0f);
			int stz = (int)glm::clamp(stx + 1.0f, 0.0f, fWidth - 1.0f);
			int stw = (int)glm::clamp(sty + 1.0f, 0.0f, fHeight - 1.0f);

			glm::vec2 t = glm::vec2(glm::clamp(pos.x - stx, 0.0f, 1.0f), glm::clamp(pos.y - sty, 0.0f, 1.0f));

			glm::vec4 v1 = m_ColorInput[stx + sty * WIDTH];
			glm::vec4 v2 = m_ColorInput[stz + sty * WIDTH];
			glm::vec4 v3 = m_ColorInput[stx + stw * WIDTH];
			glm::vec4 v4 = m_ColorInput[stz + stw * WIDTH];

			m_ColorOutput[x + y * WIDTH] = glm::lerp(glm::lerp(v1, v2, t.x), glm::lerp(v3, v4, t.x), t.y);
		}
	}
}