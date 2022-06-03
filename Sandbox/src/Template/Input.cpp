#include "stdfax.h"
#include "Input.h"

#define NUM_KEYS 1 << 9

/*
* Mapping from input-helper to the key-maps.
*/
std::map<GLFWwindow*, KeyState*> currentKeys = std::map<GLFWwindow*, KeyState*>();
std::map<GLFWwindow*, KeyState*> previousKeys = std::map<GLFWwindow*, KeyState*>();

KeyState operator&(KeyState a, KeyState b) {
	return (KeyState)((int)a & (int)b);
}
KeyState operator|(KeyState a, KeyState b) {
	return (KeyState)((int)a | (int)b);
}

/*
* Keyboard callback function.
* @param[in] window The window that received the event.
* @param[in] key that was pressed or released.
* @param[in] scancode The system-specific scancode of the key.
* @param[in] action `GLFW_PRESS`, `GLFW_RELEASE` or `GLFW_REPEAT`.
* @param[in] mods Bit field describing which modifier keys held down.
*/
void InputKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	// Key is released. 
	if (action == GLFW_RELEASE) currentKeys[window][key] = KeyState::Pressed;
	else if (action == GLFW_REPEAT) currentKeys[window][key] = KeyState::KeyDown;
	else if (action == GLFW_PRESS) currentKeys[window][key] = KeyState::KeyDown;
}

GLFWwindow* Input::s_Window = nullptr;
double Input::s_Px = 0, Input::s_Py = 0;
double Input::s_Cx = 0, Input::s_Cy = 0;
KeyState Input::s_MouseLeftPrevious = KeyState::Release, Input::s_MouseRightPrevious = KeyState::Release;
KeyState Input::s_MouseLeftCurrent = KeyState::Release, Input::s_MouseRightCurrent = KeyState::Release;

void Input::Initialize(GLFWwindow* window) {
	s_Window = window;
	// Set the callback for the input helper.
	glfwSetKeyCallback(window, InputKeyCallback);

	// Add a new mapping to the map of mappings.
	KeyState* currentKeyStates = (KeyState*)malloc(sizeof(KeyState) * NUM_KEYS);
	KeyState* previousKeyStates = (KeyState*)malloc(sizeof(KeyState) * NUM_KEYS);

	for (int i = 0; i < NUM_KEYS; i++) currentKeyStates[i] = (KeyState)GLFW_RELEASE, previousKeyStates[i] = (KeyState)GLFW_RELEASE;

	currentKeys.insert(std::pair< GLFWwindow*, KeyState*>(window, currentKeyStates));
	previousKeys.insert(std::pair< GLFWwindow*, KeyState*>(window, previousKeyStates));

	// Set the cursor position.
	glfwGetCursorPos(window, &s_Px, &s_Py);
	glfwGetCursorPos(window, &s_Cx, &s_Cy);

	s_MouseLeftCurrent = s_MouseRightCurrent = s_MouseLeftPrevious = s_MouseRightPrevious = KeyState::Release;
}

void Input::Update() {

	int* previousStates = (int*)previousKeys[s_Window];
	int* currentStates = (int*)currentKeys[s_Window];

	static const int lSize = NUM_KEYS >> 2;

	// Update the keyboard state.
	for (int i = 0; i < lSize; i++) {
		((int*)previousStates)[i] = currentStates[i];

		static const int mask = ((int)KeyState::Pressed) << 24 | ((int)KeyState::Pressed) << 16 | ((int)KeyState::Pressed) << 8 | ((int)KeyState::Pressed);
		currentStates[i] &= (currentStates[i] ^ mask);
	}

	// Update mouse position.
	s_Px = s_Cx, s_Py = s_Cy;
	glfwGetCursorPos(s_Window, &s_Cx, &s_Cy);

	// Update mouse buttons.
	s_MouseLeftPrevious = s_MouseLeftCurrent, s_MouseRightPrevious = s_MouseRightCurrent;

	int state = glfwGetMouseButton(s_Window, GLFW_MOUSE_BUTTON_LEFT);
	if (state == GLFW_PRESS) s_MouseLeftCurrent = KeyState::KeyDown;
	else {
		if ((int)(s_MouseLeftCurrent & KeyState::KeyDown)) s_MouseLeftCurrent = KeyState::Pressed;
		else s_MouseLeftCurrent = KeyState::Release;
	}

	state = glfwGetMouseButton(s_Window, GLFW_MOUSE_BUTTON_RIGHT);
	if (state == GLFW_PRESS) s_MouseRightCurrent = KeyState::KeyDown;
	else {
		if ((int)(s_MouseRightCurrent & KeyState::KeyDown)) s_MouseRightCurrent = KeyState::Pressed;
		else s_MouseRightCurrent = KeyState::Release;
	}
}

bool Input::KeyDown(Key key) {
	return (int)(previousKeys[s_Window][(int)key] & (KeyState::KeyDown | KeyState::Pressed));
}

bool Input::KeyPressed(Key key) {
	return (int)(previousKeys[s_Window][(int)key] & KeyState::Pressed);
}

bool Input::KeyUp(Key key) {
	return (int)(previousKeys[s_Window][(int)key] & KeyState::Release);
}

KeyState Input::GetKeyState(Key key) {
	return KeyState(previousKeys[s_Window][(int)key]);
}

glm::ivec2 Input::CursorPosition() {
	return glm::ivec2((int)s_Cx, (int)s_Cy);
}

glm::vec2 Input::CursorMovement() {
	return glm::vec2(s_Cx - s_Px, s_Cy - s_Py);
}

bool Input::MouseLeftButtonDown() {
	return (int)(s_MouseLeftPrevious & (KeyState::KeyDown | KeyState::Pressed));
}

bool Input::MouseLeftButtonClick() {
	return (int)(s_MouseLeftPrevious & KeyState::Pressed);
}

bool Input::MouseRightButtonDown() {
	return (int)(s_MouseRightPrevious & (KeyState::KeyDown | KeyState::Pressed));
}

bool Input::MouseRightButtonClick() {
	return (int)(s_MouseRightPrevious & KeyState::Pressed);
}

