#include "stdfax.h"
#include "Surface.h"


Surface::Surface(unsigned int width, unsigned int height)
	: m_Width(width), m_Height(height) {

	m_Pixels = (Color*)malloc(sizeof(Color) * width * height);

	// Create our render texture.
	glGenTextures(1, &m_RenderTexture);
	glBindTexture(GL_TEXTURE_2D, m_RenderTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, 0);

	glBindTexture(GL_TEXTURE_2D, 0);
	// Initialize our shader.
	m_Shader = new GLshader("simple_tex.vert", "simple_tex.frag");
	m_VertexBuffer = new GLbuffer(GL_ARRAY_BUFFER, sizeof(c_RenderQuad));
	m_VertexBuffer->Write(sizeof(c_RenderQuad), c_RenderQuad, GL_STATIC_DRAW);

	m_UVbuffer = new GLbuffer(GL_ARRAY_BUFFER, sizeof(c_UV));
	m_UVbuffer->Write(sizeof(c_UV), c_UV, GL_STATIC_DRAW);

	m_IndexBuffer = new GLbuffer(GL_ELEMENT_ARRAY_BUFFER, sizeof(c_Indices));
	m_IndexBuffer->Write(sizeof(c_Indices), c_Indices, GL_STATIC_DRAW);

	m_Shader->Activate();
	m_Shader->SetBufferFloat3(m_VertexBuffer, 0, 0);
	m_Shader->SetBufferFloat2(m_UVbuffer, 1, 0);
	m_Shader->Deactivate();

}

Surface::~Surface() {
	delete m_VertexBuffer;
	delete m_UVbuffer;
	delete m_IndexBuffer;
	delete m_Shader;
	
	free(m_Pixels);
}

void Surface::Draw() {
	m_Shader->Activate();
	glBindTexture(GL_TEXTURE_2D, m_RenderTexture);
	m_Shader->DrawTriangles(6, m_IndexBuffer, GL_UNSIGNED_INT);
	m_Shader->Deactivate();
}

void Surface::SyncPixels()
{
	glBindTexture(GL_TEXTURE_2D, m_RenderTexture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_Width, m_Height, GL_RGBA, GL_FLOAT, (GLvoid*)m_Pixels);
	glBindTexture(GL_TEXTURE_2D, 0);
	glFinish();
}

void Surface::SyncPixels(uint dx, uint dy, uint width, uint height)
{
	glBindTexture(GL_TEXTURE_2D, m_RenderTexture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, dx, dy, width, height, GL_RGBA, GL_FLOAT, (GLvoid*)m_Pixels);
	glBindTexture(GL_TEXTURE_2D, 0);
	glFinish();
}

void Surface::PlotPixel(Color color, uint x, uint y)
{
	m_Pixels[x + y * m_Width] = color;
}

void Surface::PlotPixels(Color* colors) {
	memcpy(m_Pixels, colors, sizeof(Color) * m_Width * m_Height);
}

void Surface::PlotPixels(Color* colors, uint dx, uint dy, uint width, uint height) {

	for (uint i = dy; i < dy + height; i++) {
		uint idx = dx + i * m_Width;
		memcpy(&m_Pixels[idx], &colors[idx], sizeof(Color) * width);
	}
}

