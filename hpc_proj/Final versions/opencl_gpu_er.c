#include <stdio.h>
#include <time.h>
#include <CL/cl.h>
#include <float.h>

#define NUM_RUNS 1000

#define DIM 1000

#define P 1.0

#define SEED 349872935

#define BASE 569872384

#define TS 1 // Threadblock sizes

const char *kernelstring =
    "__kernel void ER(const unsigned int N, const double p, const uint seed, const uint base,"
    "                 __global int* G){"
    "   const int globalNode_i = get_global_id(0);"
    "   const int globalNode_j = get_global_id(1);"
    "   uint seed_x = (seed + globalNode_i + globalNode_j) * (globalNode_i + 1) << globalNode_i;"
    "   uint t = seed_x ^ (seed_x << 11);"
    "   uint result = base ^ (base >> 19) ^ (t ^ (t >> 8));"
    "   double rand = (double) (result%56891279) / 56891279;"
    "   if ((rand<p) && (globalNode_i!=globalNode_j)){"
    "       G[globalNode_i*N+globalNode_j] = 1;"
    "   }"
    "   else G[globalNode_i*N+globalNode_j] = 0;"
    "}";

int main() {
    // Set the parameters
    unsigned int N = DIM; // Number of nodes
    double p = P; // Probability of edge creation
    unsigned int seed = SEED; // The seed for the OpenCL rng
    unsigned int base = BASE; // The base for the OpenCL rng
    clock_t start, end;
    double cpu_time_used;

    //Create the adjacency matrix
    int* G = (int*)malloc(N*N*sizeof(int*));

    //Configure the OpenCL environment
	printf(">>> Initializing OpenCL...\n");
    cl_platform_id platform = 0;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
    char deviceName[1024];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 1024, deviceName, NULL);
    cl_event event = NULL;

    //Compile the kernel
    cl_program program = clCreateProgramWithSource(context, 1, &kernelstring, NULL, NULL);
    clBuildProgram(program, 0, NULL, "", NULL, NULL);

    //Check for compilation errors
    size_t logSize;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    char* messages = (char*)malloc((1+logSize)*sizeof(char));
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, messages, NULL);
    messages[logSize] = '\0';
    if(logSize>10) {printf("Compiler message: %s\n", messages);}
    free(messages);

    //Prepare OpenCL memory objects
    cl_mem bufG = clCreateBuffer(context, CL_MEM_READ_WRITE, N*N*sizeof(int), NULL, NULL);

    //Copy the adjacency matrix to the GPU
    clEnqueueWriteBuffer(queue, bufG, CL_TRUE, 0, N*N*sizeof(int), G, 0, NULL, NULL);

    //Configure the kernel and set its arguments
    cl_kernel kernel = clCreateKernel(program, "ER", NULL);
    clSetKernelArg(kernel, 0, sizeof(unsigned int), (void*)&N);
    clSetKernelArg(kernel, 1, sizeof(double), (void*)&p);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), (void*)&seed);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), (void*)&base);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufG);

    //Start the timed loop
    printf(">>> Starting %d ER runs...\n", NUM_RUNS);
    start = clock();
    for (int r=0; r<NUM_RUNS; r++) {
        // Run the kernel
        const size_t local[2] = { TS, TS };
        const size_t global[2] = { N, N };
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);

        // Wait for calculations to be finished
        clWaitForEvents(1, &event);
    }

    // End the timed loop
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    cpu_time_used /= NUM_RUNS;
    int Digs = DECIMAL_DIG;
    printf("Mean time taken to create the graph: %.*e seconds\n", Digs, cpu_time_used);

    // Copy the output back to CPU memory
    clEnqueueReadBuffer(queue, bufG, CL_TRUE, 0, N*N*sizeof(int), G, 0, NULL, NULL);

    /*printf("Adjacency Matrix:\n");
    for (int i = 0; i < (int)N; i++) {
        for (int j = 0; j < (int)N; j++) {
            printf("%d ", G[i*N+j]);
        }
        printf("\n");
    }*/

    // Free the OpenCL memory objects
    clReleaseMemObject(bufG);

    // Clean-up OpenCL
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    // Free the host memory objects
    free(G);

    return 0;
}
