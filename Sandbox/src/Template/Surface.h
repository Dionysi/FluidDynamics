#pragma once
#include "Shader.h"

/*
* Surface used for rendering to the screen.An OpenGL texture is used to render to a quad.
*/
class Surface {

public:

	/*
	* Initialize the surface according to the specified size.
	* @param[in] width		Surface width.
	* @param[in] height		Surface height.
	*/
	Surface(unsigned int width, unsigned int height);
	~Surface();

	/*
	* Renders the render texture to the current OpenGL context.
	*/
	void Draw();

	/*
	* Synchronize the entire pixel array to the GPU.
	*/
	void SyncPixels();
	/*
	* Synchronize the defined area to the GPU.
	* @param[in] colors		Colors to write to the surface.
	* @param[in] dx			x-offset.
	* @param[in] dy			y-offset.
	* @param[in] width		Number of pixels in x-direction.
	* @param[in] height		Number of pixels in y-direction.
	*/
	void SyncPixels(uint dx, uint dy, uint width, uint height);

	/*
	* Plot a single pixel.
	* @param[in] color		Pixel color.
	* @param[in] x			x-coordinate.
	* @param[in] y			y-coordinate.
	*/
	void PlotPixel(Color color, uint x, uint y);
	/*
	* Plots the pixel array to the surface. <b>Note:</b> must be of size width * height.
	* @param[in] colors		Array of size width * height.
	*/
	void PlotPixels(Color* colors);
	/*
	* Write pixels to the surface.
	* @param[in] colors		Colors to write to the surface.
	* @param[in] dx			x-offset.
	* @param[in] dy			y-offset.
	* @param[in] width		Number of pixels in x-direction.
	* @param[in] height		Number of pixels in y-direction.
	*/
	void PlotPixels(Color* colors, uint dx, uint dy, uint width, uint height);

	inline GLuint& GetRenderTexture() { return m_RenderTexture; }
	inline unsigned int GetWidth() { return m_Width; }
	inline unsigned int GetHeight() { return m_Height; }

private:
	/*
	* Surface dimensions.
	*/
	unsigned int m_Width, m_Height;

	/*
	* Shader used for rendering.
	*/
	GLshader* m_Shader = nullptr;
	// OpenGL buffers.
	GLbuffer* m_VertexBuffer = nullptr;
	GLbuffer* m_UVbuffer = nullptr;
	GLbuffer* m_IndexBuffer = nullptr;

	/*
	* Texture used for rendering.
	*/
	GLuint m_RenderTexture;

	/*
	* Quad used for rendering the texture to.
	*/
	const GLfloat c_RenderQuad[18] = {
		-1.0f, -1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
		1.0f,  1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f
	};
	/*
	* UV-coordinates for sampling the render texture.
	*/
	const GLfloat c_UV[12] = {
		0.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 1.0f,
		1.0f, 1.0f,
		1.0f, 0.0f,
		0.0f, 0.0f
	};
	/*
	* Indices for drawing triangles.
	*/
	const GLuint c_Indices[6] = {
		0, 1, 2, 3, 4, 5
	};

	/*
	* Array containing our CPU pixel data.
	*/
	Color* m_Pixels = nullptr;
};