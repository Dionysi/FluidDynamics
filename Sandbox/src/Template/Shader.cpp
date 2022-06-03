#include "stdfax.h"
#include "Shader.h"

#include <glm/gtc/type_ptr.hpp>

#pragma region General
GLuint CreateProgramFromSource(const char* vertShaderSrc, const char* fragShaderSrc) {
	GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);

	GLint result = GL_FALSE;
	int logLength;

	// Compile vertex shader
	glShaderSource(vertShader, 1, &vertShaderSrc, NULL);
	glCompileShader(vertShader);

	// Check vertex shaders
	glGetShaderiv(vertShader, GL_COMPILE_STATUS, &result);
	glGetShaderiv(vertShader, GL_INFO_LOG_LENGTH, &logLength);
	std::vector<char> vertShaderError((logLength > 1) ? logLength : 1);
	glGetShaderInfoLog(vertShader, logLength, NULL, &vertShaderError[0]);
	if (logLength > 0)
		std::cout << &vertShaderError[0] << std::endl;

	// Compile fragment shader
	glShaderSource(fragShader, 1, &fragShaderSrc, NULL);
	glCompileShader(fragShader);

	// Check fragment shader
	glGetShaderiv(fragShader, GL_COMPILE_STATUS, &result);
	glGetShaderiv(fragShader, GL_INFO_LOG_LENGTH, &logLength);
	std::vector<char> fragShaderError((logLength > 1) ? logLength : 1);
	glGetShaderInfoLog(fragShader, logLength, NULL, &fragShaderError[0]);
	if (logLength > 0)
		std::cout << &fragShaderError[0] << std::endl;

	GLuint program = glCreateProgram();
	glAttachShader(program, vertShader);
	glAttachShader(program, fragShader);
	glLinkProgram(program);

	glGetProgramiv(program, GL_LINK_STATUS, &result);
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
	std::vector<char> programError((logLength > 1) ? logLength : 1);
	glGetProgramInfoLog(program, logLength, NULL, &programError[0]);
	if (logLength > 0) FATAL_ERROR("Failed to create create shader: %s", programError.data());

	glDeleteShader(vertShader);
	glDeleteShader(fragShader);

	return program;
}

GLuint CreateProgramFromPath(const char* vertexPath, const char* fragmentPath) {

	// Read shaders
	std::string vertShaderStr = readFile(vertexPath);
	std::string fragShaderStr = readFile(fragmentPath);
	const char* vertShaderSrc = vertShaderStr.c_str();
	const char* fragShaderSrc = fragShaderStr.c_str();

	return CreateProgramFromSource(vertShaderSrc, fragShaderSrc);
}
#pragma endregion

#pragma region Buffer

GLbuffer::GLbuffer(GLenum type, size_t size) : m_Type(type) {
	glGenBuffers(1, &m_Buffer);								// Create the buffer.
	glBindBuffer(m_Type, m_Buffer);
	glBufferData(m_Type, size, NULL, GL_STATIC_DRAW);		// Reserve memory on the gpu.
	glBindBuffer(m_Type, 0);
}

GLbuffer::~GLbuffer() {
	glDeleteBuffers(1, &m_Buffer);
}

void GLbuffer::Bind() { glBindBuffer(m_Type, m_Buffer); }
void GLbuffer::Unbind() { glBindBuffer(m_Type, 0); }

void GLbuffer::Write(size_t size, const void* src, GLenum usage) {
	glBindBuffer(m_Type, m_Buffer);
	glBufferData(m_Type, size, src, usage);
	glBindBuffer(m_Type, 0);
}
void GLbuffer::Write(size_t size, size_t offset, const void* src) {
	glBindBuffer(m_Type, m_Buffer);
	glBufferSubData(m_Buffer, offset, size, src);
	glBindBuffer(m_Type, 0);
}

#pragma endregion

#pragma region Shader
GLshader::GLshader(const char* vertexPath, const char* fragmentPath) {
	m_Program = CreateProgramFromPath(vertexPath, fragmentPath);

	glUseProgram(m_Program);
	glGenVertexArrays(1, &m_VAO);
}
GLshader::GLshader(std::string vertexSource, std::string fragmentSource) {
	m_Program = CreateProgramFromSource(vertexSource.c_str(), fragmentSource.c_str());

	glUseProgram(m_Program);
	glGenVertexArrays(1, &m_VAO);
}
GLshader::~GLshader() {
	glDeleteVertexArrays(1, &m_VAO);
	glDeleteProgram(m_Program);
}

void GLshader::SetBufferFloat1(GLbuffer* buffer, uint idx, size_t stride) {
	glBindVertexArray(m_VAO);
	glEnableVertexAttribArray(idx);
	buffer->Bind();
	glVertexAttribPointer(idx, 1, GL_FLOAT, GL_FALSE, stride, (void*)0);
	glBindVertexArray(0);
	glDisableVertexAttribArray(idx);
}
void GLshader::SetBufferFloat2(GLbuffer* buffer, uint idx, size_t stride) {
	buffer->Bind();
	glEnableVertexAttribArray(idx);
	glVertexAttribPointer(idx, 2, GL_FLOAT, GL_FALSE, stride, (void*)0);
}
void GLshader::SetBufferFloat3(GLbuffer* buffer, uint idx, size_t stride) {
	buffer->Bind();
	glEnableVertexAttribArray(idx);
	glVertexAttribPointer(idx, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
}
void GLshader::SetBufferFloat4(GLbuffer* buffer, uint idx, size_t stride) {
	glBindVertexArray(m_VAO);
	glEnableVertexAttribArray(idx);
	buffer->Bind();
	glVertexAttribPointer(idx, 4, GL_FLOAT, GL_FALSE, stride, (void*)0);
	glBindVertexArray(0);
	glDisableVertexAttribArray(idx);
}

void GLshader::SetBufferInt1(GLbuffer* buffer, uint idx, size_t stride) {
	glBindVertexArray(m_VAO);
	glEnableVertexAttribArray(idx);
	buffer->Bind();
	glVertexAttribPointer(idx, 1, GL_INT, GL_FALSE, stride, (void*)0);
	glBindVertexArray(0);
	glDisableVertexAttribArray(idx);
}
void GLshader::SetBufferInt2(GLbuffer* buffer, uint idx, size_t stride) {
	glBindVertexArray(m_VAO);
	glEnableVertexAttribArray(idx);
	buffer->Bind();
	glVertexAttribPointer(idx, 2, GL_INT, GL_FALSE, stride, (void*)0);
	glBindVertexArray(0);
	glDisableVertexAttribArray(idx);
}
void GLshader::SetBufferInt3(GLbuffer* buffer, uint idx, size_t stride) {
	glBindVertexArray(m_VAO);
	glEnableVertexAttribArray(idx);
	buffer->Bind();
	glVertexAttribPointer(idx, 3, GL_INT, GL_FALSE, stride, (void*)0);
	glBindVertexArray(0);
	glDisableVertexAttribArray(idx);
}
void GLshader::SetBufferInt4(GLbuffer* buffer, uint idx, size_t stride) {
	glBindVertexArray(m_VAO);
	glEnableVertexAttribArray(idx);
	buffer->Bind();
	glVertexAttribPointer(idx, 4, GL_INT, GL_FALSE, stride, (void*)0);
	glBindVertexArray(0);
	glDisableVertexAttribArray(idx);
}

void GLshader::SetBufferUint1(GLbuffer* buffer, uint idx, size_t stride) {
	glBindVertexArray(m_VAO);
	glEnableVertexAttribArray(idx);
	buffer->Bind();
	glVertexAttribPointer(idx, 1, GL_UNSIGNED_INT, GL_FALSE, stride, (void*)0);
	glBindVertexArray(0);
	glDisableVertexAttribArray(idx);
}
void GLshader::SetBufferUint2(GLbuffer* buffer, uint idx, size_t stride) {
	glBindVertexArray(m_VAO);
	glEnableVertexAttribArray(idx);
	buffer->Bind();
	glVertexAttribPointer(idx, 2, GL_UNSIGNED_INT, GL_FALSE, stride, (void*)0);
	glBindVertexArray(0);
	glDisableVertexAttribArray(idx);
}
void GLshader::SetBufferUint3(GLbuffer* buffer, uint idx, size_t stride) {
	glBindVertexArray(m_VAO);
	glEnableVertexAttribArray(idx);
	buffer->Bind();
	glVertexAttribPointer(idx, 3, GL_UNSIGNED_INT, GL_FALSE, stride, (void*)0);
	glBindVertexArray(0);
	glDisableVertexAttribArray(idx);
}
void GLshader::SetBufferUint4(GLbuffer* buffer, uint idx, size_t stride) {
	glBindVertexArray(m_VAO);
	glEnableVertexAttribArray(idx);
	buffer->Bind();
	glVertexAttribPointer(idx, 4, GL_UNSIGNED_INT, GL_FALSE, stride, (void*)0);
	glBindVertexArray(0);
	glDisableVertexAttribArray(idx);
}

void GLshader::SetUniformFloat(const char* name, float val) {
	glUseProgram(m_Program);
	glUniform1f(glGetUniformLocation(m_Program, name), val);
	glUseProgram(0);
}
void GLshader::SetUniformVec2(const char* name, glm::vec2 val) {
	glUseProgram(m_Program);
	glUniform2fv(glGetUniformLocation(m_Program, name), 1, glm::value_ptr(val));
	glUseProgram(0);
}
void GLshader::SetUniformVec3(const char* name, glm::vec3 val) {
	glUseProgram(m_Program);
	glUniform3fv(glGetUniformLocation(m_Program, name), 1, glm::value_ptr(val));
	glUseProgram(0);
}
void GLshader::SetUniformVec4(const char* name, glm::vec4 val) {
	glUseProgram(m_Program);
	glUniform4fv(glGetUniformLocation(m_Program, name), 1, glm::value_ptr(val));
	glUseProgram(0);
}
void GLshader::SetUniformInt(const char* name, int val) {
	glUseProgram(m_Program);
	glUniform1i(glGetUniformLocation(m_Program, name), val);
	glUseProgram(0);
}
void GLshader::SetUniformVec2i(const char* name, glm::ivec2 val) {
	glUseProgram(m_Program);
	glUniform2iv(glGetUniformLocation(m_Program, name), 1, glm::value_ptr(val));
	glUseProgram(0);
}
void GLshader::SetUniformVec3i(const char* name, glm::ivec3 val) {
	glUseProgram(m_Program);
	glUniform2iv(glGetUniformLocation(m_Program, name), 1, glm::value_ptr(val));
	glUseProgram(0);
}
void GLshader::SetUniformVec4i(const char* name, glm::ivec4 val) {
	glUseProgram(m_Program);
	glUniform2iv(glGetUniformLocation(m_Program, name), 1, glm::value_ptr(val));
	glUseProgram(0);
}
void GLshader::SetUniformMat4(const char* name, glm::mat4 mat) {
	glUseProgram(m_Program);
	glUniformMatrix4fv(glGetUniformLocation(m_Program, name), 1, GL_FALSE, &mat[0][0]);
	glUseProgram(0);
}

void GLshader::Activate() {
	glUseProgram(m_Program);
	glBindVertexArray(m_VAO);
}
void GLshader::Deactivate() {
	glBindVertexArray(0);
	glUseProgram(0);
}

void GLshader::DrawLines(size_t count, GLbuffer* indexBuffer, GLenum idxType) {

	indexBuffer->Bind();									// Bind index buffer.
	glDrawElements(GL_LINES, count, idxType, (void*)0);		// Draw call.
}
void GLshader::DrawTriangles(size_t count, GLbuffer* indexBuffer, GLenum idxType) {

	indexBuffer->Bind();									// Bind index buffer.
	glDrawElements(GL_TRIANGLES, count, idxType, NULL);		// Draw call.
}

#pragma endregion

