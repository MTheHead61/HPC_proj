#include <stdio.h>
#include <time.h>
#include <CL/cl.h>
#include <float.h>

#define NUM_RUNS 1000

#define DIM 1000

#define R 1.0

#define SEED 349872935

#define TS 1 // Threadblock sizes

const char *kernelstring =
    "__kernel void RGG(const unsigned int N, const double dist,"
    "   const __global double* X, const __global double* Y,"
    "   __global int* G){"
    "   const int globalNode_i = get_global_id(0);"
    "   const int globalNode_j = get_global_id(1);"
    "   if (globalNode_i != globalNode_j) {"
    "       double x = X[globalNode_i] - X[globalNode_j];"
    "       double y = Y[globalNode_i] - Y[globalNode_j];"
    "       double r = x*x + y*y;"
    "       r = sqrt(r);"
    "       if (r < dist) {"
    "           G[globalNode_i*N+globalNode_j] = 1;"
    "           G[globalNode_j*N+globalNode_i] = 1;"
    "       }"
    "   }"
    "}";

int main() {
    // Set the parameters
    int N = DIM; // Number of nodes
    double dist = R;
    int i;
    double* X = (double*)malloc(N*sizeof(double*)); // Array with the nodes abscissa
    double* Y = (double*)malloc(N*sizeof(double*)); // Array with the nodes ordinate
    clock_t start, end;
    double cpu_time_used;

    //Create the adjacency matrix
    int* G = (int*)malloc(N*N*sizeof(int*));

    // Set the random seed
    srand(SEED);

    // Create random N points in a 2-D space
    for (i=0; i<N; i++){
        X[i] = (double)rand()/RAND_MAX; // We get a random number in interval [0,1[
        Y[i] = (double)rand()/RAND_MAX;
    }

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
    cl_mem bufX = clCreateBuffer(context, CL_MEM_READ_ONLY, N*sizeof(double), NULL, NULL);
    cl_mem bufY = clCreateBuffer(context, CL_MEM_READ_ONLY, N*sizeof(double), NULL, NULL);

    //Copy the adjacency matrix to the GPU
    clEnqueueWriteBuffer(queue, bufG, CL_TRUE, 0, N*N*sizeof(int), G, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0, N*sizeof(double), X, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufY, CL_TRUE, 0, N*sizeof(double), Y, 0, NULL, NULL);

    //Configure the kernel and set its arguments
    cl_kernel kernel = clCreateKernel(program, "RGG", NULL);
    clSetKernelArg(kernel, 0, sizeof(unsigned int), (void*)&N);
    clSetKernelArg(kernel, 1, sizeof(double), (void*)&dist);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufX);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufY);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufG);

    //Start the timed loop
    printf(">>> Starting %d RGG runs...\n", NUM_RUNS);
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
