#include "stdfax.h"
#include <stdarg.h>     /* va_list, va_start, va_arg, va_end */
#include <fstream>



void FATAL_ERROR(const char* format, ...) {
	char buffer[8192];
	va_list args;
	va_start(args, format);
	vsnprintf(buffer, sizeof(buffer), format, args);
	va_end(args);
	std::cerr << buffer << std::endl;
	exit(0);
}

std::string readFile(const char* filePath) {

	std::string content;
	std::ifstream fileStream(filePath, std::ios::in);
	

	if (!fileStream.is_open()) {
		std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
		return "";
	}

	std::string line = "";
	while (!fileStream.eof()) {
		std::getline(fileStream, line);
		content.append(line + "\n");
	}

	fileStream.close();
	return content;
}


#pragma region GLdebug

const char* DebugTypeToString(GLenum type) {
	switch (type) {
	case GL_DEBUG_TYPE_ERROR:
		return "ERROR";
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
		return "DEPRECATED BEHAVIOR";
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
		return "UDEFINED BEHAVIOR";
	case GL_DEBUG_TYPE_PORTABILITY:
		return "PORTABILITY";
	case GL_DEBUG_TYPE_PERFORMANCE:
		return "PERFORMANCE";
	case GL_DEBUG_TYPE_OTHER:
		return "OTHER";
	case GL_DEBUG_TYPE_MARKER:
		return "MARKER";
	default:
		return "UNKNOWN";
	}
}

const char* DebugSeverityToString(GLenum severity) {
	switch (severity) {
	case GL_DEBUG_SEVERITY_HIGH:
		return "HIGH";
	case GL_DEBUG_SEVERITY_MEDIUM:
		return "MEDIUM";
	case GL_DEBUG_SEVERITY_LOW:
		return "LOW";
	case GL_DEBUG_SEVERITY_NOTIFICATION:
		return "NOTIFICATION";
	default:
		return "UNKNOWN";
	}
}

/* Source: https://www.khronos.org/opengl/wiki/OpenGL_Error */
void GLAPIENTRY MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
	GLsizei length, const GLchar* message, const void* userParam) {

	if (type != GL_DEBUG_TYPE_ERROR) return;
	FATAL_ERROR("GL ERROR: Severity = %s\n%s", DebugSeverityToString(severity), message);
}

void EnableGLdebugInfo() {
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(MessageCallback, 0);
}

void DisableGLdebugInfo() {
	glDisable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(nullptr, 0);
}
#pragma endregion

#pragma region Color

Color Color::operator/(float rhs) {
	return Color(r / rhs, g / rhs, b / rhs, a / rhs);
}
Color Color::operator/=(float rhs) {
	r /= rhs, g /= rhs, b /= rhs, a /= rhs;
	return *this;
}
Color Color::operator*(float rhs) {
	return Color(r * rhs, g * rhs, b * rhs, a * rhs);
}
Color Color::operator*=(float rhs) {
	r *= rhs, g *= rhs, b *= rhs, a *= rhs;
	return *this;
}
Color Color::operator+(float rhs) {
	return Color(r + rhs, g + rhs, b + rhs, a + rhs);
}
Color Color::operator+=(float rhs) {
	r += rhs, g += rhs, b += rhs, a += rhs;
	return *this;
}
Color Color::operator-(float rhs) {
	return Color(r - rhs, g - rhs, b - rhs, a - rhs);
}
Color Color::operator-=(float rhs) {
	r -= rhs, g -= rhs, b -= rhs, a -= rhs;
	return *this;
}

Color Color::operator/(Color rhs) {
	return Color(r / rhs.r, g / rhs.g, b / rhs.b, a / rhs.a);
}
Color Color::operator/=(Color rhs) {
	r /= rhs.r, g /= rhs.g, b /= rhs.b, a /= rhs.a;
	return *this;
}
Color Color::operator*(Color rhs) {
	return Color(r * rhs.r, g * rhs.g, b * rhs.b, a * rhs.a);
}
Color Color::operator*=(Color rhs) {
	r *= rhs.r, g *= rhs.g, b *= rhs.b, a *= rhs.a;
	return *this;
}
Color Color::operator+(Color rhs) {
	return Color(r + rhs.r, g + rhs.g, b + rhs.b, a + rhs.a);
}
Color Color::operator+=(Color rhs) {
	r += rhs.r, g += rhs.g, b += rhs.b, a += rhs.a;
	return *this;
}
Color Color::operator-(Color rhs) {
	return Color(r - rhs.r, g - rhs.g, b - rhs.b, a - rhs.a);
}
Color Color::operator-=(Color rhs) {
	r -= rhs.r, g -= rhs.g, b -= rhs.b, a -= rhs.a;
	return *this;
}

#pragma endregion


/*
* Array containing human-friendly names for OpenCL error codes.
* Source: https://github.com/martijnberger/clew/blob/master/src/clew.c
*/
const char* errorCodeStrings[69] = {
	// Error Codes
	  "CL_SUCCESS"                                  //   0
	, "CL_DEVICE_NOT_FOUND"                         //  -1
	, "CL_DEVICE_NOT_AVAILABLE"                     //  -2
	, "CL_COMPILER_NOT_AVAILABLE"                   //  -3
	, "CL_MEM_OBJECT_ALLOCATION_FAILURE"            //  -4
	, "CL_OUT_OF_RESOURCES"                         //  -5
	, "CL_OUT_OF_HOST_MEMORY"                       //  -6
	, "CL_PROFILING_INFO_NOT_AVAILABLE"             //  -7
	, "CL_MEM_COPY_OVERLAP"                         //  -8
	, "CL_IMAGE_FORMAT_MISMATCH"                    //  -9
	, "CL_IMAGE_FORMAT_NOT_SUPPORTED"               //  -10
	, "CL_BUILD_PROGRAM_FAILURE"                    //  -11
	, "CL_MAP_FAILURE"                              //  -12
	, "CL_MISALIGNED_SUB_BUFFER_OFFSET"             //  -13
	, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"//  -14
	, "CL_COMPILE_PROGRAM_FAILURE"                  //  -15
	, "CL_LINKER_NOT_AVAILABLE"                     //  -16
	, "CL_LINK_PROGRAM_FAILURE"                     //  -17
	, "CL_DEVICE_PARTITION_FAILED"                  //  -18
	, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"            //  -19

	, ""    //  -20
	, ""    //  -21
	, ""    //  -22
	, ""    //  -23
	, ""    //  -24
	, ""    //  -25
	, ""    //  -26
	, ""    //  -27
	, ""    //  -28
	, ""    //  -29

	, "CL_INVALID_VALUE"                            //  -30
	, "CL_INVALID_DEVICE_TYPE"                      //  -31
	, "CL_INVALID_PLATFORM"                         //  -32
	, "CL_INVALID_DEVICE"                           //  -33
	, "CL_INVALID_CONTEXT"                          //  -34
	, "CL_INVALID_QUEUE_PROPERTIES"                 //  -35
	, "CL_INVALID_COMMAND_QUEUE"                    //  -36
	, "CL_INVALID_HOST_PTR"                         //  -37
	, "CL_INVALID_MEM_OBJECT"                       //  -38
	, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"          //  -39
	, "CL_INVALID_IMAGE_SIZE"                       //  -40
	, "CL_INVALID_SAMPLER"                          //  -41
	, "CL_INVALID_BINARY"                           //  -42
	, "CL_INVALID_BUILD_OPTIONS"                    //  -43
	, "CL_INVALID_PROGRAM"                          //  -44
	, "CL_INVALID_PROGRAM_EXECUTABLE"               //  -45
	, "CL_INVALID_KERNEL_NAME"                      //  -46
	, "CL_INVALID_KERNEL_DEFINITION"                //  -47
	, "CL_INVALID_KERNEL"                           //  -48
	, "CL_INVALID_ARG_INDEX"                        //  -49
	, "CL_INVALID_ARG_VALUE"                        //  -50
	, "CL_INVALID_ARG_SIZE"                         //  -51
	, "CL_INVALID_KERNEL_ARGS"                      //  -52
	, "CL_INVALID_WORK_DIMENSION"                   //  -53
	, "CL_INVALID_WORK_GROUP_SIZE"                  //  -54
	, "CL_INVALID_WORK_ITEM_SIZE"                   //  -55
	, "CL_INVALID_GLOBAL_OFFSET"                    //  -56
	, "CL_INVALID_EVENT_WAIT_LIST"                  //  -57
	, "CL_INVALID_EVENT"                            //  -58
	, "CL_INVALID_OPERATION"                        //  -59
	, "CL_INVALID_GL_OBJECT"                        //  -60
	, "CL_INVALID_BUFFER_SIZE"                      //  -61
	, "CL_INVALID_MIP_LEVEL"                        //  -62
	, "CL_INVALID_GLOBAL_WORK_SIZE"                 //  -63
	, "CL_INVALID_PROPERTY"                         //  -64
	, "CL_INVALID_IMAGE_DESCRIPTOR"                 //  -65
	, "CL_INVALID_COMPILER_OPTIONS"                 //  -66
	, "CL_INVALID_LINKER_OPTIONS"                   //  -67
	, "CL_INVALID_DEVICE_PARTITION_COUNT"           //  -68
};

bool CL_ERROR(cl_int error, const char* msg) {
	//#ifdef _DEBUG
#if 1
	if (error == CL_SUCCESS) return true;

	static const int num_errors = sizeof(errorCodeStrings) / sizeof(errorCodeStrings[0]);

	if (error == -1001) {
		return "CL_PLATFORM_NOT_FOUND_KHR";
	}

	if (error > 0 || -error >= num_errors) {
		return "Unknown OpenCL error";
	}

	FATAL_ERROR("Application terminated with OpenCL error code (%i): %s\n", error, errorCodeStrings[-error]);
#else
	return true;
#endif	
}

double GetGPUProfilingTimeInformation(gpu_event pEvent, GPU_PROFILING_COMMAND pInfoType) {
	cl_ulong time;
	clGetEventProfilingInfo(pEvent, (cl_profiling_info)pInfoType, sizeof(cl_ulong), &time, NULL);
	return time;
}

double GetGPUCommandExecutionTime(gpu_event pEvent) {
	cl_ulong startTime, endTime;
	clGetEventProfilingInfo(pEvent, (cl_profiling_info)GPU_PROFILING_COMMAND::START, sizeof(cl_ulong), &startTime, NULL);
	clGetEventProfilingInfo(pEvent, (cl_profiling_info)GPU_PROFILING_COMMAND::END, sizeof(cl_ulong), &endTime, NULL);
	return (endTime - startTime) * 1e-6; // convert from ns to ms
}

void SetEventCallback(gpu_event gEvent,
	GPU_EVENT_TYPE gEventType,
	void(CL_CALLBACK* callback)(gpu_event gEvent, cl_int event_command_status, void* user_data),
	void* userData) {

	CL_ERROR(
		clSetEventCallback(gEvent, (cl_int)gEventType, callback, userData),
		"Failed to set event callback."
	);
}


#pragma region Context
clContext::clContext(bool glInteropEnabled) {
	GetPlatformAndDevice();
	CreateContext(glInteropEnabled);
}

clContext::~clContext() {
	CL_ERROR(clReleaseDevice(m_DeviceID), "failed to release device");
	CL_ERROR(clReleaseContext(m_Context), "failed to release context");
}

void clContext::GetPlatformAndDevice() {
	// Retrieve all platforms for the system.
	cl_uint platformCount;

	(clGetPlatformIDs(0, NULL, &platformCount), "unable to retrieve platforms");

	cl_platform_id* platforms = (cl_platform_id*)malloc(platformCount * sizeof(cl_platform_id*));
	CL_ERROR(clGetPlatformIDs(platformCount, platforms, NULL), "unable to retrieve platforms");

	char pInfo[512];
	// Retrieve the first Nvidia or AMD GPU that we can find.
	for (cl_uint i = 0; i < platformCount; i++) {
		if (CL_ERROR(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 512, &pInfo, NULL), "unable to retrieve platform info"))
			if (strstr(pInfo, "NVIDIA") || strstr(pInfo, "AMD"))
				if (CL_ERROR(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &m_DeviceID, NULL), "unable to retrieve devices")) {
					m_PlatformID = platforms[i];
					free(platforms);
					return;
				}
	}

	free(platforms);

	FATAL_ERROR("Unable to find a suitable GPU.");
}

void clContext::CreateContext(bool glInteropEnabled) {
	cl_context_properties properties[]{
			CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
				CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
				CL_CONTEXT_PLATFORM, (cl_context_properties)m_PlatformID,
				0
	};

	cl_int errorCode;
	m_Context = clCreateContext(glInteropEnabled ? properties : nullptr, 1, &m_DeviceID, NULL, NULL, &errorCode);
	CL_ERROR(errorCode, "could not create cl_context");
}

void clContext::PrintDeviceInfo() {
	char vendor[256];				// Vendor.
	char name[256];					// Name.
	char deviceVersion[256];				// Driver version
	cl_bool	available;				// Availability.
	cl_device_type deviceType;		// Type.

	cl_uint coreCount;				// Amount of compute units.
	size_t maxWarpSize;				// Maximum work-group size.
	cl_uint clockFrequency;			// Clock frequency.

	cl_long memorySize;				// Size of the (global) memory.
	cl_ulong maxAllocatableMemory;	// Maximum allocatable memory.
	cl_ulong localMemorySize;		// Size of the local memory.
	cl_uint cacheLineSize;			// Cache line size.
	cl_ulong cacheSize;				// Cache size.
	cl_device_mem_cache_type cacheType; // Cache type.

	cl_bool imageSupport;
	size_t maxHeight, maxWidth, maxCount;

	clGetDeviceInfo(m_DeviceID, CL_DEVICE_NAME, sizeof(name), name, NULL);
	clGetDeviceInfo(m_DeviceID, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
	clGetDeviceInfo(m_DeviceID, CL_DEVICE_VERSION, sizeof(deviceVersion), deviceVersion, NULL);
	clGetDeviceInfo(m_DeviceID, CL_DEVICE_AVAILABLE, sizeof(available), &available, NULL);
	clGetDeviceInfo(m_DeviceID, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);

	clGetDeviceInfo(m_DeviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(coreCount), &coreCount, NULL);
	clGetDeviceInfo(m_DeviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWarpSize), &maxWarpSize, NULL);
	clGetDeviceInfo(m_DeviceID, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockFrequency), &clockFrequency, NULL);

	clGetDeviceInfo(m_DeviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memorySize), &memorySize, NULL);
	clGetDeviceInfo(m_DeviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxAllocatableMemory), &maxAllocatableMemory, NULL);
	clGetDeviceInfo(m_DeviceID, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemorySize), &localMemorySize, NULL);
	clGetDeviceInfo(m_DeviceID, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cacheLineSize), &cacheLineSize, NULL);
	clGetDeviceInfo(m_DeviceID, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cacheSize), &cacheSize, NULL);
	clGetDeviceInfo(m_DeviceID, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(cacheType), &cacheType, NULL);

	clGetDeviceInfo(m_DeviceID, CL_DEVICE_IMAGE_SUPPORT, sizeof(imageSupport), &imageSupport, NULL);
	clGetDeviceInfo(m_DeviceID, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(maxWidth), &maxWidth, NULL);
	clGetDeviceInfo(m_DeviceID, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(maxHeight), &maxHeight, NULL);
	clGetDeviceInfo(m_DeviceID, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, sizeof(maxCount), &maxCount, NULL);


	printf("============= Device info: =============\n");
	printf("\tGeneral:\n");
	printf("\t\tName:\t\t\t\t%s\n", name);
	printf("\t\tVendor:\t\t\t\t%s\n", vendor);
	printf("\t\tVersion:\t\t\t%s\n", deviceVersion);
	printf("\t\tAvailable:\t\t\t%s\n", available ? "Yes" : "No");
	printf("\t\tDevice Type Flags:\t\t%s%s%s%s\n", deviceType & CL_DEVICE_TYPE_CPU ? "CPU " : "", deviceType & CL_DEVICE_TYPE_GPU ? "GPU " : "", deviceType & CL_DEVICE_TYPE_ACCELERATOR ? "Accelerator " : "", deviceType & CL_DEVICE_TYPE_DEFAULT ? "Default " : "");

	printf("\tComputing Power:\n");
	printf("\t\tCompute Units:\t\t\t%u\n", coreCount);
	printf("\t\tMax. Work-Group Size:\t\t%zu\n", maxWarpSize);
	printf("\t\tClock Frequency:\t\t%u mHz\n", clockFrequency);

	printf("\tMemory:\n");
	printf("\t\tGlobal Memory:\t\t\t%0.00f MB\n", (double)memorySize / 1048576.0);
	printf("\t\tMax. Allocatable Memory:\t%0.00f MB\n", (double)maxAllocatableMemory / 1048576.0);
	printf("\t\tLocal Memory:\t\t\t%0.00f KB\n", (double)localMemorySize / 1024.0);
	printf("\t\tCache Line Size:\t\t%u B\n", cacheLineSize);
	printf("\t\tCache Type:\t\t\t%s\n", cacheType == CL_READ_ONLY_CACHE ? "Read-Only" : cacheType == CL_READ_WRITE_CACHE ? "Read-Write" : "None");
	if (cacheType != CL_NONE)
		printf("\t\tCache Size:\t\t\t%0.00f KB\n", (double)cacheSize / 1024.0);

	printf("\tImages:\n");
	printf("\t\tImage support:\t\t\t%s\n", imageSupport ? "Yes" : "No");
	printf("\t\tMax. Image2D size:\t\t(%i, %i)\n", maxWidth, maxHeight);
	printf("\t\tMax. image array size:\t\t%i\n", maxCount);

	printf("\n");
}

#pragma endregion

#pragma region Program
clProgram::clProgram(clContext* context, const char* path) {

	CreateProgram(context, path);
	BuildProgram(context);
}

clProgram::~clProgram() {
	CL_ERROR(clReleaseProgram(m_Program), "failed to release program");
}

void clProgram::CreateProgram(clContext* context, const char* path) {
	// Load and build our OpenCL source code.
	const char* source;
	size_t* sizes = (size_t*)malloc(sizeof(size_t) * 1);
	source = ReadSource(path, sizes);

	// Error code.
	cl_int errorCode;
	// Try to create the program.
	m_Program = clCreateProgramWithSource(context->GetContext(), 1, &source, sizes, &errorCode);

	// Check for errors.
	CL_ERROR(errorCode, "could not create cl program.");
}

void clProgram::BuildProgram(clContext* context) {
	cl_int buildStatus =
		clBuildProgram(m_Program, 1, &context->GetDeviceID(),
			"-cl-fast-relaxed-math -cl-mad-enable -cl-denorms-are-zero -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only",
			NULL, NULL
		);

	// Check for errors during building.
	if (buildStatus != CL_SUCCESS) {
		char* log = new char[100 * 1024]; // Error log can be pretty large.
		log[0] = 0;
		clGetProgramBuildInfo(m_Program, context->GetDeviceID(), CL_PROGRAM_BUILD_LOG, 100 * 1024, log, NULL);
		log[2048] = 0; // Truncate very long logs.
		printf("\n%s\n", log);

		FATAL_ERROR("Failed to build the OpenCL program:\n%s\n", log);
	}
}

char* clProgram::ReadSource(const char* filePath, size_t* size) {
	std::string source;
	// extract path from source file name
	char path[2048];
	strcpy_s(path, filePath);
	char* marker = path, * fileName = (char*)filePath;
	while (strstr(marker + 1, "\\")) marker = strstr(marker + 1, "\\");
	while (strstr(marker + 1, "/")) marker = strstr(marker + 1, "/");
	while (strstr(fileName + 1, "\\")) fileName = strstr(fileName + 1, "\\");
	while (strstr(fileName + 1, "/")) fileName = strstr(fileName + 1, "/");
	if (fileName != filePath) fileName++;
	*marker = 0;
	// load source file
	FILE* f;
	fopen_s(&f, filePath, "r");
	if (!f) throw std::exception("Error loading source");
	char line[8192];
	int lineNr = 0;
	while (!feof(f)) {
		line[0] = 0;
		fgets(line, 8190, f);
		lineNr++;
		// clear source file line
		while (line[0]) {
			if (line[strlen(line) - 1] > 32) break;
			line[strlen(line) - 1] = 0;
		}
		// expand error commands
		char* err = strstr(line, "Error(");
		if (err) {
			char rem[8192], cmd[128];
			strcpy_s(rem, err + 6);
			*err = 0;
			sprintf_s(cmd, "Error_( %i, %i,", 0, lineNr);
			strcat_s(line, cmd);
			strcat_s(line, rem);
		}
		// expand assets
		char* as = strstr(line, "Assert(");
		if (as) {
			char rem[8192], cmd[128];
			strcpy_s(rem, as + 7);
			*as = 0;
			sprintf_s(cmd, "Assert_( %i, %i,", 0, lineNr);
			strcat_s(line, cmd);
			strcat_s(line, rem);
		}
		// handle include files
		char* inc = strstr(line, "#include");
		if (inc) {
			char* start = strstr(inc, "\"");
			if (!start) throw std::exception("Preprocessor error in #include statement line");
			char* end = strstr(start + 1, "\"");
			if (!end) throw std::exception("Preprocessor error in #include statement line");
			char file[2048];
			*end = 0;
			strcpy_s(file, path);
			strcat_s(file, "/");
			strcat_s(file, start + 1);
			char* incText = ReadSource(file, size);
			source.append(incText);
		}
		else {
			source.append(line);
			source.append("\n");
		}
	}
	*size = strlen(source.c_str());
	char* t = (char*)malloc(*size + 1);
	strcpy_s(t, INT32_MAX, source.c_str());
	fclose(f);
	return t;
}
#pragma endregion

#pragma region Command Queue
clCommandQueue::clCommandQueue(clContext* context, bool outOfOrderEnabled, bool profilingEnabled) {
	// Construct the command queue properties.
	cl_command_queue_properties pOutOfOrder = outOfOrderEnabled ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0;
	cl_command_queue_properties pProfiling = profilingEnabled ? CL_QUEUE_PROFILING_ENABLE : 0;

	cl_int errorCode;
	m_Queue = clCreateCommandQueue(context->GetContext(), context->GetDeviceID(), pOutOfOrder | pProfiling, &errorCode);
	CL_ERROR(errorCode, "Failed to create command queue");
}

clCommandQueue::~clCommandQueue() {
	CL_ERROR(clReleaseCommandQueue(m_Queue), "Failed to release command queue");
}

void clCommandQueue::Synchronize() {
	CL_ERROR(clFinish(m_Queue), "Failed to synchronize the command queue.");
}

void clCommandQueue::Flush() {
	CL_ERROR(clFlush(m_Queue), "Failed to flush the command queue.");
}
#pragma endregion

#pragma region Buffer
clBuffer::clBuffer(clContext* context, size_t size, BufferFlags flags) : m_BufferSize(size) {

	cl_int errorCode;
	m_Buffer = clCreateBuffer(context->GetContext(), cl_mem_flags(flags), size, nullptr, &errorCode);
	CL_ERROR(errorCode, "Failed to create buffer.");
}

clBuffer::clBuffer(clContext* context, unsigned int glTexture) : m_BufferSize(0) {
	cl_int errorCode;
	m_Buffer = clCreateFromGLTexture(context->GetContext(), CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, glTexture, &errorCode);
	CL_ERROR(errorCode, "Failed to create buffer from glTexture.");
}

clBuffer::clBuffer(clContext* context, clCommandQueue* queue, size_t size, void*& dataPtr, bool writeOnly) : m_BufferSize(size) {

	cl_int errorCode;
	m_Buffer = clCreateBuffer(context->GetContext(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, NULL, &errorCode);
	CL_ERROR(errorCode, "Failed to create pinned memory buffer.");
	dataPtr = clEnqueueMapBuffer(queue->GetCommandQueue(), m_Buffer, CL_TRUE, writeOnly ? CL_MAP_WRITE : CL_MAP_READ, 0, size, 0, NULL, NULL, &errorCode);
	CL_ERROR(errorCode, "Failed to create map memory buffer.");
}

clBuffer::clBuffer(clContext* context, clCommandQueue* queue, cl_image_format* format, cl_image_desc* desc, void*& dataPtr, bool writeOnly)
	: m_BufferSize(0), m_Desc(desc), m_Format(format) {
	cl_int errorCode;
	m_Buffer = clCreateImage(context->GetContext(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, format, desc, NULL, &errorCode);
	CL_ERROR(errorCode, "Failed to create pinned gpu image buffer.");
	size_t origin[3]{ 0, 0, 0 };
	size_t region[3]{ desc->image_width, desc->image_height, desc->image_depth };
	dataPtr = clEnqueueMapImage(queue->GetCommandQueue(), m_Buffer, CL_TRUE, writeOnly ? CL_MAP_WRITE : CL_MAP_READ, origin, region, &desc->image_row_pitch, &desc->image_slice_pitch, 0, NULL, NULL, &errorCode);
	CL_ERROR(errorCode, "Failed to create map gpu image buffer.");
}

clBuffer::clBuffer(clContext* context, clCommandQueue* queue, cl_image_format* format, cl_image_desc* desc)
	: m_BufferSize(0), m_Desc(desc), m_Format(format) {

	cl_int errorCode;
	m_Buffer = clCreateImage(context->GetContext(), CL_MEM_READ_WRITE, format, desc, NULL, &errorCode);
	CL_ERROR(errorCode, "Failed to create gpu image buffer.");
}

clBuffer::~clBuffer() {
	CL_ERROR(clReleaseMemObject(m_Buffer), "Failed to release buffer.");
}

void clBuffer::CopyToDevice(clCommandQueue* queue, void* src, bool blocking, gpu_event* pEvent) {
	CL_ERROR(
		clEnqueueWriteBuffer(queue->GetCommandQueue(), m_Buffer, blocking, 0, m_BufferSize, src, 0, nullptr, pEvent),
		"Failed to copy data to device buffer."
	);
}

void clBuffer::CopyToDevice(clCommandQueue* queue, void* src, size_t offset, size_t size, bool blocking, gpu_event* pEvent) {
	CL_ERROR(
		clEnqueueWriteBuffer(queue->GetCommandQueue(), m_Buffer, blocking, offset, size, src, 0, nullptr, pEvent),
		"Failed to copy data to device buffer."
	);
}

void clBuffer::CopyToHost(clCommandQueue* queue, void* dst, bool blocking, gpu_event* pEvent) {
	CL_ERROR(
		clEnqueueReadBuffer(queue->GetCommandQueue(), m_Buffer, blocking, 0, m_BufferSize, dst, 0, nullptr, pEvent),
		"Failed to copy data to device buffer."
	);
}

void clBuffer::CopyToHost(clCommandQueue* queue, void* dst, size_t offset, size_t size, bool blocking, gpu_event* pEvent) {
	CL_ERROR(
		clEnqueueReadBuffer(queue->GetCommandQueue(), m_Buffer, blocking, offset, size, dst, 0, nullptr, pEvent),
		"Failed to copy data to device buffer."
	);
}

void clBuffer::CopyToDeviceImage(clCommandQueue* queue, void* src, bool blocking, gpu_event* pEvent) {
	if (!m_Format || !m_Desc) FATAL_ERROR("clBuffer is not an OpenCL image object (CopyToDeviceImage).");

	static size_t origin[3]{ 0, 0, 0 };
	size_t region[3]{ m_Desc->image_width , m_Desc->image_height, m_Desc->image_depth };

	CL_ERROR(
		clEnqueueWriteImage(queue->GetCommandQueue(), m_Buffer, blocking, origin, region, 0, 0, src, 0, NULL, pEvent),
		"Failed to copy data to device image."
	);
}

void clBuffer::CopyToDeviceImage(clCommandQueue* queue, void* src, size_t origin[3], size_t region[3], bool blocking, gpu_event* pEvent) {
	if (!m_Format || !m_Desc) FATAL_ERROR("clBuffer is not an OpenCL image object (CopyToDeviceImage).");
	CL_ERROR(
		clEnqueueWriteImage(queue->GetCommandQueue(), m_Buffer, blocking, origin, region, m_Desc->image_row_pitch, m_Desc->image_slice_pitch, src, 0, NULL, pEvent),
		"Failed to copy data to device image."
	);
}

void clBuffer::CopyToHostImage(clCommandQueue* queue, void* dst, bool blocking, gpu_event* pEvent) {
	if (!m_Format || !m_Desc) FATAL_ERROR("clBuffer is not an OpenCL image object (CopyToHostImage).");

	static size_t origin[3]{ 0, 0, 0 };
	size_t region[3]{ m_Desc->image_width , m_Desc->image_height, m_Desc->image_depth };

	CL_ERROR(
		clEnqueueReadImage(queue->GetCommandQueue(), m_Buffer, blocking, origin, region, 0, 0, dst, 0, NULL, pEvent),
		"Failed to copy data from device image."
	);
}

void clBuffer::CopyToHostImage(clCommandQueue* queue, void* dst, size_t origin[3], size_t region[3], bool blocking, gpu_event* pEvent) {
	if (!m_Format || !m_Desc) FATAL_ERROR("clBuffer is not an OpenCL image object (CopyToHostImage).");

	CL_ERROR(
		clEnqueueReadImage(queue->GetCommandQueue(), m_Buffer, blocking, origin, region, 0, 0, dst, 0, NULL, pEvent),
		"Failed to copy data from device image."
	);
}

void clBuffer::CopyBufferToImage(clCommandQueue* queue, clBuffer* buffer, clBuffer* image, size_t imgDims[3], gpu_event* pEvent) {

	static const size_t origin[3] = { 0, 0, 0 };
	CL_ERROR(
		clEnqueueCopyBufferToImage(queue->GetCommandQueue(), buffer->GetBuffer(), image->GetBuffer(), 0, origin, imgDims, 0, NULL, pEvent),
		"Failed to copy buffer to image."
	);
}

void clBuffer::CopyBufferToImage(clCommandQueue* queue, clBuffer* buffer, clBuffer* image, size_t srcOffset, size_t dstOrigin[3], size_t dstRegion[3], gpu_event* pEvent) {
	CL_ERROR(
		clEnqueueCopyBufferToImage(queue->GetCommandQueue(), buffer->GetBuffer(), image->GetBuffer(), srcOffset, dstOrigin, dstRegion, 0, NULL, pEvent),
		"Failed to copy buffer to image."
	);
}

void clBuffer::CopyImageToBuffer(clCommandQueue* queue, clBuffer* image, clBuffer* buffer, size_t imgDims[3], gpu_event* pEvent) {
	static const size_t origin[3] = { 0, 0, 0 };
	CL_ERROR(
		clEnqueueCopyImageToBuffer(queue->GetCommandQueue(), image->GetBuffer(), buffer->GetBuffer(), origin, imgDims, 0, 0, NULL, pEvent),
		"Failed to copy image to buffer."
	);
}

void clBuffer::CopyImageToBuffer(clCommandQueue* queue, clBuffer* image, clBuffer* buffer, size_t srcOrigin[3], size_t srcRegion[3], size_t dstOrigin, gpu_event* pEvent) {
	CL_ERROR(
		clEnqueueCopyImageToBuffer(queue->GetCommandQueue(), image->GetBuffer(), buffer->GetBuffer(), srcOrigin, srcRegion, dstOrigin, 0, NULL, pEvent),
		"Failed to copy image to buffer."
	);
}

void clBuffer::AcquireGLObject(clCommandQueue* queue, gpu_event* pEvent) {
	CL_ERROR(
		clEnqueueAcquireGLObjects(queue->GetCommandQueue(), 1, &m_Buffer, 0, NULL, pEvent),
		"Failed to Acquire GL object."
	);
}

void clBuffer::ReleaseGLObject(clCommandQueue* queue, gpu_event* pEvent) {
	CL_ERROR(
		clEnqueueReleaseGLObjects(queue->GetCommandQueue(), 1, &m_Buffer, 0, NULL, pEvent),
		"Failed to Release GL object."
	);
}

void clBuffer::MapImage(clCommandQueue* queue, void*& dataPtr, bool writeOnly, gpu_event* pEvent) {
	cl_int errorCode;
	size_t origin[3]{ 0, 0, 0 };
	size_t region[3]{ m_Desc->image_width, m_Desc->image_height, m_Desc->image_depth };
	dataPtr = clEnqueueMapImage(queue->GetCommandQueue(), m_Buffer, CL_TRUE, writeOnly ? CL_MAP_WRITE : CL_MAP_READ, origin, region, &m_Desc->image_row_pitch, &m_Desc->image_slice_pitch, 0, NULL, NULL, &errorCode);
	CL_ERROR(errorCode, "Failed to create map gpu image buffer.");
}

void clBuffer::UnmapBuffer(clCommandQueue* queue, void* dataPtr, gpu_event* pEvent) {
	cl_int errorCode;
	clEnqueueUnmapMemObject(queue->GetCommandQueue(), m_Buffer, dataPtr, 0, NULL, pEvent);
}
#pragma endregion

#pragma region Kernel
clKernel::clKernel(clProgram* program, const char* kernelName) {
	cl_int errorCode;
	m_Kernel = clCreateKernel(program->GetProgram(), kernelName, &errorCode);
	CL_ERROR(errorCode, "Failed to create kernel.");
}

clKernel::~clKernel() {
	CL_ERROR(clReleaseKernel(m_Kernel), "Failed to release kernel.");
}

void clKernel::SetArgument(unsigned int index, void* arg, size_t size) {
	CL_ERROR(clSetKernelArg(m_Kernel, index, size, arg), "Failed to set kernel argument");
}

void clKernel::SetArgument(unsigned int index, clBuffer* buffer) {
	CL_ERROR(clSetKernelArg(m_Kernel, index, sizeof(cl_mem), &(buffer->GetBuffer())), "Failed to set kernel argument");
}

void clKernel::Enqueue(clCommandQueue* queue, size_t globalSize, size_t localSize, gpu_event* pEvent) {
	CL_ERROR(
		clEnqueueNDRangeKernel(queue->GetCommandQueue(), m_Kernel, 1, NULL, &globalSize, &localSize, 0, nullptr, pEvent),
		"Failed to enqueue kernel."
	);
}

void clKernel::Enqueue(clCommandQueue* queue, unsigned int workDim, size_t* globalWorkSize, size_t* localWorkSize, gpu_event* pEvent) {
	CL_ERROR(
		clEnqueueNDRangeKernel(queue->GetCommandQueue(), m_Kernel, workDim, NULL, globalWorkSize, localWorkSize, 0, nullptr, pEvent),
		"Failed to enqueue kernel."
	);
}
#pragma endregion