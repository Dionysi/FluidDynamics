#pragma once
#define NOMINMAX

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <bitset>

#include <glew/glew.h>
#include <glfw/glfw3.h>

#include <imgui_impl_opengl3.h>
#include <imgui_impl_glfw.h>

#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <Windows.h>


typedef uint8_t uchar;
typedef uint16_t ushort;
typedef uint32_t uint;
typedef uint64_t ulong;

void FATAL_ERROR(const char* format, ...);

/*
* Reads the contents of a file.
* @param[in] Path to the file.
* @returns Contents of the file.
*/
std::string readFile(const char* filePath);

#pragma region GLdebug
void EnableGLdebugInfo();
void DisableGLdebugInfo();
#pragma endregion

struct Color {
	Color() : r(0.0f), g(0.0f), b(0.0f), a(0.0f) {}
	Color(float val) : r(val), g(val), b(val), a(1.0f) {}
	Color(float r, float g, float b) : r(r), g(g), b(b), a(1.0f) {}
	Color(float r, float g, float b, float a) : r(r), g(g), b(b), a(a) {}

	Color operator/(float rhs);
	Color operator/=(float rhs);
	Color operator*(float rhs);
	Color operator*=(float rhs);
	Color operator+(float rhs);
	Color operator+=(float rhs);
	Color operator-(float rhs);
	Color operator-=(float rhs);

	Color operator/(Color rhs);
	Color operator/=(Color rhs);
	Color operator*(Color rhs);
	Color operator*=(Color rhs);
	Color operator+(Color rhs);
	Color operator+=(Color rhs);
	Color operator-(Color rhs);
	Color operator-=(Color rhs);

	float r, g, b, a;
};

#pragma region OpenCL
typedef cl_event gpu_event;

enum class GPU_PROFILING_COMMAND {
	/* Identifies when the command was queued by the host. */
	QUEUED = CL_PROFILING_COMMAND_QUEUED,
	/* Identifies when the command was submitted by the host. */
	SUBMIT = CL_PROFILING_COMMAND_SUBMIT,
	/* Identifies when the command started executing on the device. */
	START = CL_PROFILING_COMMAND_START,
	/* Identifies when the command has finished executing on the device. */
	COMPLETE = CL_PROFILING_COMMAND_COMPLETE,
	/* Identifies when the command and all of its child commands finished executing on the device. */
	END = CL_PROFILING_COMMAND_END
};

enum class GPU_EVENT_TYPE {
	SUBMITTED = CL_SUBMITTED,
	RUNNING = CL_RUNNING,
	COMPLETE = CL_COMPLETE
};


/*
* Checks for OpenCL errors.
* @param[in] error		OpenCL error code.
* @param[in] msg		Optional error message.
* @returns				true if CL_SUCCESS.
*/
bool CL_ERROR(cl_int error, const char* msg);

/*
* Retrieve the profiling information for a given profiling event.
* @param[in] pEvent			A valid gpu profiling event.
* @param[in] pInfoType		Type of profiling information to retrieve.
* @returns					Profiling time-measure.
*/
double GetGPUProfilingTimeInformation(gpu_event pEvent, GPU_PROFILING_COMMAND pInfoType);

/*
* Retrieves the time it took for a command to start executing until completion.
* @param[in] pEvent			Valid profiling event.
* @returns					Execution time in ms.
*/
double GetGPUCommandExecutionTime(gpu_event pEvent);

/*
* Sets a callback for a gpu event.
* @param[in] gEvent			gpu_event for which to set the callback.
* @param[in] gEventType		Type of GPU event for which the callback is triggered.
* @param[in] callback		Callback function pointer.
* @param[in] userData		Pointer with data to pass to the callback function.
*/
void SetEventCallback(gpu_event gEvent,
	GPU_EVENT_TYPE gEventType,
	void (CL_CALLBACK* callback)(gpu_event gEvent, cl_int event_command_status, void* user_data),
	void* userData);

enum class BufferFlags {
	/** This flag specifies that the memory object is a readonly memory object when used inside a kernel. */
	READ_ONLY = CL_MEM_READ_ONLY,
	/** This flag specifies that the memory object will be written but not read by a kernel. */
	WRITE_ONLY = CL_MEM_WRITE_ONLY,
	/** This flag specifies that the memory object will be readand written by a kernel.This is the default. */
	READ_WRITE = CL_MEM_READ_WRITE,
};

class clContext {

public:
	/*
	* Creates an OpenCL context.
	* @param[in] glInteropEnabled		Indicates whether gl-cl interop should be enabled.
	*/
	clContext(bool glInteropEnabled);
	~clContext();


	/*
	* Prints the device info associated with the device for this opencl program.
	*/
	void PrintDeviceInfo();

	/*
	* Retrieves the device id.
	*/
	const cl_device_id& GetDeviceID() { return m_DeviceID; }
	/*
	* Retrieves the OpenCL context.
	*/
	const cl_context& GetContext() { return m_Context; }

private:
	cl_platform_id m_PlatformID = 0;
	cl_device_id m_DeviceID = 0;
	cl_context m_Context = 0;

	/*
	* Retrieves the most optimal to use platform and device.
	*/
	void GetPlatformAndDevice();
	/*
	* Create an OpenCL command queue for the context and device.
	* @param[in] glInteropEnabled		Indicates whether gl-cl interop should be enabled.
	*/
	void CreateContext(bool glInteropEnabled);
};

class clProgram {

public:
	/*
	* Creates an OpenCL context and constructs a program from the provided path.
	* @param[in] context	Valid OpenCL context.
	* @param[in] path		Path to the OpenCL source code.
	*/
	clProgram(clContext* context, const char* path);
	~clProgram();

	/*
	* Retrieves the OpenCL program.
	*/
	const cl_program& GetProgram() { return m_Program; }

private:

	cl_program m_Program = 0;

	/*
	* Create a cl_program form the given source.
	* @param[in] context	Valid OpenCL context.
	* @param[in] path		Path to the OpenCL source code.
	*/
	void CreateProgram(clContext* context, const char* path);
	/*
	* Builds the cl_program.
	* @param[in] context	Valid OpenCL context.
	*/
	void BuildProgram(clContext* context);
	/*
	* Reads an OpenCL file and does some funky pre-processing. No idea where this method came from.
	* @param[in] filePath		File path to the OpenCL file.
	* @param[out] size			Pointer to where the size of the string should be stored. Cannot be NULL. I think.
	* @returns					Array with the source code of the OpenCL program.
	*/
	char* ReadSource(const char* filePath, size_t* size);
};

class clCommandQueue {

public:
	/*
	* Creates an OpenCL command queue.
	* @param[in] context				Valid OpenCL context.
	* @param[in] outOfOrderEnabled		Determines whether the commands queued in the command-queue are executed in-order or out-of-order.
	* @param[in] profilingEnabled		Enable or disable profiling of commands in the command-queue.
	*/
	clCommandQueue(clContext* context, bool outOfOrderEnabled = false, bool profilingEnabled = false);
	~clCommandQueue();

	/*
	* Blocks untill all previously queued commands on the queue are issued to the associated device and have completed.
	*/
	void Synchronize();
	/*
	* Issues all previously queued commands to the device associated with the command queue.
	*/
	void Flush();

	/*
	* Retrieves the OpenCL command queue.
	* @returns OpenCL command queue.
	*/
	const cl_command_queue& GetCommandQueue() { return m_Queue; }

private:
	cl_command_queue m_Queue = 0;
};

class clBuffer {

public:
	/*
	* Creates an OpenCL memory buffer.
	* @param[in] context		Valid OpenCL context.
	* @param[in] size			Size of the buffer in bytes.
	* @param[in] flags			Flags specifying use of the memory by OpenCL.
	*/
	clBuffer(clContext* context, size_t size, BufferFlags flags);
	/*
	* Construct a buffer object from an OpenGL texture.
	* @param[in] context		Valid OpenCL context.
	* @param[in] glTexture			Valid OpenGL texture.
	*/
	clBuffer(clContext* context, unsigned int glTexture);
	/*
	* Creates a memory buffer and allocates pinned memory.
	* @param[in]  context		Valid OpenCL context.
	* @param[in]  queue			Command queue used to create the pinned memory.
	* @param[in]  size			Size of the buffer in bytes.
	* @param[out] dataPtr		Pointer where the data will be stored.
	* @param[in] writeOnly		Indicates whether the mapped region is used for writing or reading by the host.
	*/
	clBuffer(clContext* context, clCommandQueue* queue, size_t size, void*& dataPtr, bool writeOnly = true);
	/*
	* Creates an OpenCL image and maps its content data on the host.
	* @param[in]  context		Valid OpenCL context.
	* @param[in]  queue			Command queue used to create the pinned memory.
	* @param[in]  format		Image format.
	* @param[in]  desc			Image description.
	* @param[out] dataPtr		Pointer where the data will be stored.
	* @param[in] writeOnly		Indicates whether the mapped region is used for writing or reading by the host.
	*/
	clBuffer(clContext* context, clCommandQueue* queue, cl_image_format* format, cl_image_desc* desc, void*& dataPtr, bool writeOnly = true);
	/*
	* Creates an OpenCL image.
	* @param[in]  context		Valid OpenCL context.
	* @param[in]  queue			Command queue used to create the pinned memory.
	* @param[in]  format		Image format.
	* @param[in]  desc			Image description.
	*/
	clBuffer(clContext* context, clCommandQueue* queue, cl_image_format* format, cl_image_desc* desc);

	~clBuffer();

	/*
	* Copy data from the host to the device buffer.
	* @param[in] queue			Valid command queue.
	* @param[in] src			Source buffer on host.
	* @param[in] blocking		Blocking-write if set to true.
	* @param[out] pEvent		Profiling event used for retrieving profiling data.
	*/
	void CopyToDevice(clCommandQueue* queue, void* src, bool blocking = true, gpu_event* pEvent = NULL);
	/*
	* Copy data from the host to the device buffer.
	* @param[in] queue			Valid command queue.
	* @param[in] src			Source buffer on host.
	* @param[in] offset			Offset in bytes.
	* @param[in] size			Size to copy in bytes.
	* @param[in] blocking		Blocking-write if set to true.
	* @param[out] pEvent		Profiling event used for retrieving profiling data.
	*/
	void CopyToDevice(clCommandQueue* queue, void* src, size_t offset, size_t size, bool blocking = true, gpu_event* pEvent = NULL);

	/*
	* Copy data from the device buffer to the host.
	* @param[in] queue			Valid command queue.
	* @param[in] dst			Destination buffer on host.
	* @param[in] blocking		Blocking-read if set to true.
	* @param[out] pEvent		Profiling event used for retrieving profiling data.
	*/
	void CopyToHost(clCommandQueue* queue, void* dst, bool blocking = true, gpu_event* pEvent = NULL);
	/*
	* Copy data from the device buffer to the host.
	* @param[in] queue			Valid command queue.
	* @param[in] dst			Destination buffer on host.
	* @param[in] offset			Offset in bytes.
	* @param[in] size			Size to copy in bytes.
	* @param[in] blocking		Blocking-read if set to true.
	* @param[out] pEvent		Profiling event used for retrieving profiling data.
	*/
	void CopyToHost(clCommandQueue* queue, void* dst, size_t offset, size_t size, bool blocking = true, gpu_event* pEvent = NULL);

	/*
	* Copy data from the host to the device image.
	* @param[in] queue			Valid command queue.
	* @param[in] src			Source buffer on host.
	* @param[in] blocking		Blocking-write if set to true.
	* @param[out] pEvent		Profiling event used for retrieving profiling data.
	*/
	void CopyToDeviceImage(clCommandQueue* queue, void* src, bool blocking = true, gpu_event* pEvent = NULL);
	/*
	* Copy data from the host to the device image.
	* @param[in] queue			Valid command queue.
	* @param[in] src			Source buffer on host.
	* @param[in] origin			Origin in x, y, z from where the device will start writing. y and z should be 0 if not 2D or 3D images respectively.
	* @param[in] region			Region in width, height, depth that the device will write. Height and depth should be 1 if not 2D or 3D images respectively.
	* @param[in] blocking		Blocking-write if set to true.
	* @param[out] pEvent		Profiling event used for retrieving profiling data.
	*/
	void CopyToDeviceImage(clCommandQueue* queue, void* src, size_t origin[3], size_t region[3], bool blocking = true, gpu_event* pEvent = NULL);

	/*
	* Copy data from the device image to the host.
	* @param[in] queue			Valid command queue.
	* @param[in] dst			Destination buffer on host.
	* @param[in] blocking		Blocking-write if set to true.
	* @param[out] pEvent		Profiling event used for retrieving profiling data.
	*/
	void CopyToHostImage(clCommandQueue* queue, void* dst, bool blocking = true, gpu_event* pEvent = NULL);
	/*
	* Copy data from the device image to the host.
	* @param[in] queue			Valid command queue.
	* @param[in] dst			Destination buffer on host.
	* @param[in] origin			Origin in x, y, z from where the device will start reading. y and z should be 0 if not 2D or 3D images respectively.
	* @param[in] region			Region in width, height, depth that the device will read. Height and depth should be 1 if not 2D or 3D images respectively.
	* @param[in] blocking		Blocking-write if set to true.
	* @param[out] pEvent		Profiling event used for retrieving profiling data.
	*/
	void CopyToHostImage(clCommandQueue* queue, void* dst, size_t origin[3], size_t region[3], bool blocking = true, gpu_event* pEvent = NULL);

	/*
	* Copies all the contents of the buffer to the image.
	* @param[in] queue				A valid command queue used to queue the operation on.
	* @param[in] buffer				Source buffer.
	* @param[in] image				Destination image.
	* @param[in] imgDims			3-item array specifying the dimensions of the image.
	*/
	static void CopyBufferToImage(clCommandQueue* queue, clBuffer* buffer, clBuffer* image, size_t imgDims[3], gpu_event* pEvent = NULL);
	/*
	* Copies all the contents of the buffer to the 3D image.
	* @param[in] queue				A valid command queue used to queue the operation on.
	* @param[in] buffer				Source buffer.
	* @param[in] image				Destination image.
	* @param[in] srcOffset			Offset in the source buffer.
	* @param[in] dstOrigin			Origin in x,y,z in the image of where to start writing.
	* @param[in] dstRegion			Destination rectangle in width, height, depth (x,y,z) in which the data is written.
	* @param[out] pEvent			Profiling event used for retrieving profiling data.
	*/
	static void CopyBufferToImage(clCommandQueue* queue, clBuffer* buffer, clBuffer* image, size_t srcOffset, size_t dstOrigin[3], size_t dstRegion[3], gpu_event* pEvent = NULL);

	/*
	* Copies all the contents of the 3D image to .
	* @param[in] queue				A valid command queue used to queue the operation on.
	* @param[in] image				Source image.
	* @param[in] buffer				Destination buffer.
	* @param[in] imgDims			3-item array specifying the dimensions of the image.
	*/
	static void CopyImageToBuffer(clCommandQueue* queue, clBuffer* image, clBuffer* buffer, size_t imgDims[3], gpu_event* pEvent = NULL);
	/*
	* Copies all the contents of the 3D image to .
	* @param[in] queue				A valid command queue used to queue the operation on.
	* @param[in] image				Source image.
	* @param[in] buffer				Destination buffer.
	* @param[in] srcOrigin			Origin in x,y,z in the image of where to start reading.
	* @param[in] srcRegion			Source rectangle in width, height, depth (x,y,z) in which the data is read.
	* @param[in] dstOffset			Offset in the destination buffer.
	* @param[out] pEvent			Profiling event used for retrieving profiling data.
	*/
	static void CopyImageToBuffer(clCommandQueue* queue, clBuffer* image, clBuffer* buffer, size_t srcOrigin[3], size_t srcRegion[3], size_t dstOrigin, gpu_event* pEvent = NULL);
	/*
	* Device-to-device copy. 
	* @param[in] queue		Valid command queue.
	* @param[in] dst		Destination device-buffer.
	* @param[in] src		Source device-buffer.
	* @param[in] size		Number of bytes to copy.
	* @param[in] offset		Offset from where the copy starts (same for src and dst buffers).
	* @param[out] pEvent			Profiling event used for retrieving profiling data.
	*/
	static void CopyBufferToBuffer(clCommandQueue* queue, clBuffer* dst, clBuffer* src, size_t size, size_t offset = 0, gpu_event* pEvent = NULL);

	/*
	* Acquire opengl texture. <b>NOTE:</b> buffer should be constructed from an OpenGL texture.
	* @param[in] queue		Valid OpenCL command queue.
	* @param[out] pEvent		Profiling event used for retrieving profiling data.
	*/
	void AcquireGLObject(clCommandQueue* queue, gpu_event* pEvent = NULL);
	/*
	* Release OpenGL objects. <b>NOTE:</b> buffer should be constructed from an OpenGL texture.
	* @param[in] queue		Valid OpenCL command queue.
	* @param[out] pEvent		Profiling event used for retrieving profiling data.
	*/
	void ReleaseGLObject(clCommandQueue* queue, gpu_event* pEvent = NULL);


	void MapImage(clCommandQueue* queue, void*& dataPtr, bool writeOnly = true, gpu_event* pEvent = NULL);
	/*
	* Unmaps a region of pinned memory.
	* @param[in] queue			Command queue used to perform the unmap operation on.
	* @param[in] dataPtr		Pointer to the pinned memory region. Must be previously pinned by buffer object.
	* @param[out] pEvent		Profiling event used for retrieving profiling data.
	*/
	void UnmapBuffer(clCommandQueue* queue, void* dataPtr, gpu_event* pEvent = NULL);

	/*
	* Retrieve the OpenCL memory object.
	*/
	const cl_mem& GetBuffer() { return m_Buffer; }
	/*
	* Retrieves te buffer size.
	*/
	const size_t GetSize() { return m_BufferSize; }

private:
	cl_mem m_Buffer = 0;
	/*
	* Buffer size in bytes. */
	size_t m_BufferSize;

	/* Image descriptor if the buffer is an OpenCL image.*/
	cl_image_desc* m_Desc = nullptr;
	/* Image format if the buffer is an OpenCL image. */
	cl_image_format* m_Format = nullptr;

};

class clKernel {

public:
	/*
	* Creates an OpenCL kernel.
	* @param[in] program		Valid OpenCL program.
	* @param[in] kernelName		Name of the kernel.
	*/
	clKernel(clProgram* program, const char* kernelName);
	~clKernel();

	/*
	* Set a kernel argument.
	* @param[in] index			Index of the argument to set.
	* @param[in] arg			Pointer to the argument data.
	* @param[in] size			Size of the parameter.
	*/
	void SetArgument(unsigned int index, void* arg, size_t size);
	/*
	* Set a kernel argument.
	* @param[in] index			Index of the argument to set.
	* @param[in] buffer			Valid buffer object.
	*/
	void SetArgument(unsigned int index, clBuffer* buffer);

	/*
	* Enqueues a kernel for execution.
	* @param[in] queue					Valid command queue.
	* @param[in] globalWorkSize			Number of total threads.
	* @param[in] localWorkSize			Number of threads in a local group. <b>NOTE!</b>  globalSizeshould be a multiple of localSize.
	* @param[out] pEvent				Profiling event used for retrieving profiling data.
	*/
	void Enqueue(clCommandQueue* queue, size_t globalSize, size_t localSize, gpu_event* pEvent = NULL);
	/*
	* Enqueues a kernel for execution.
	* @param[in] queue					Valid command queue.
	* @param[in] workDim				Work dimensions.
	* @param[in] globalWorkSize			Number of total threads, must be of length workDim.
	* @param[in] localWorkSize			Number of threads in a local group, must be of length workDim. <b>NOTE!</b>  globalSize[0], ..., globalSize[workDim - 1] should be a multiple of localSize[0], ..., localSize[workDim - 1] respectively.
	* @param[out] pEvent				Profiling event used for retrieving profiling data.
	*/
	void Enqueue(clCommandQueue* queue, unsigned int workDim, size_t* globalWorkSize, size_t* localWorkSize, gpu_event* pEvent = NULL);

private:
	cl_kernel m_Kernel = 0;

};
#pragma endregion