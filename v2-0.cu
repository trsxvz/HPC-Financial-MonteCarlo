#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>



/* ---------- BASIC CUDA FUNCTIONS ---------- */



// Function that catches the CUDA error 
void testCUDA(cudaError_t error, const char* file, int line) {
	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d : %s\n", file, line, cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

// Function to print the current GPU specs and properties (testing purposes)
void printMainDeviceProperties() {
    int deviceID; cudaGetDevice(&deviceID);
    cudaDeviceProp prop;cudaGetDeviceProperties(&prop, deviceID);
    printf("\n===== Main CUDA GPU device %d: %s =====\n", deviceID, prop.name);
    printf("  Total global memory : %lu bytes\n", prop.totalGlobalMem);
    printf("  Max threads per block : %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads per multiprocessor : %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Number of streaming multiprocessors : %d\n", prop.multiProcessorCount);
    printf("  Max grid size : (%d, %d, %d)\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Warp size : %d\n", prop.warpSize);
    printf("  Max simultaneous threads : %d\n", 
           prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount);
    printf("===========================\n");
}



/* ---------- CUDA MONTE-CARLO KERNEL ---------- */



// CUDA device kernel for Monte Carlo simulation
__global__ void monte_carlo_kernel(float *results, int N, int steps, float sqrtdt, float S0, float Y0, float K, float C1, float C2, float C3, float C4) {
    
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    // Setup random number generator for this thread
    curandState state;
    curand_init((unsigned long long)clock64() + idx, idx, 0, &state);

    // Each thread processes multiple trajectories in increments of totalThreads
    if (idx < N) {

        // Initial conditions
        float S = S0, Y = Y0;

        //Random variables
        float2 randPair;
        float G1, G2;

        // Perform the simulation
        for (int j = 0; j < steps; j++) {
            randPair = curand_normal2(&state);
			G1 = randPair.x;
            G2 = randPair.y;
            S = S * (1.0f + expf(Y) * sqrtdt * G1);
            Y = C1 * Y + C2 - C3 * G1 + C4 * G2 ;
        }

        // Compute payoff and store in global memory
        results[idx] = S-K > 0.0f ? S-K: 0.0f;
    }
}



/* ---------- CUDA REDUCTION KERNEL ---------- */



// CUDA kernel for sum reduction
__global__ void reduction_kernel(float* input, float* output, int n) {
    
    // Declaring a shared memory array for the block
    extern __shared__ float sharedData[];

    // Calculating thread ID and block ID
    int idx = threadIdx.x;
    int globalIndex = blockIdx.x * blockDim.x + idx;

    // Loading data into shared memory
    sharedData[idx] = (globalIndex < n) ? input[globalIndex] : 0.0f;
    __syncthreads();

    // Performing reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (idx < stride) {
            sharedData[idx] += sharedData[idx + stride];
        }
        __syncthreads();
    }

    // Writing the block result to the output array
    if (idx == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}



/* ---------- MAIN FUNCTION ---------- */



int main(int argc, char** argv) {

    // Model parameters
    float S0 = 1.0f, Y0 = logf(0.1f);
    float K = 1.0f, T = 1.0f;
    float dt = 0.001f;
    int steps = (int)(T/dt);
    int N;
    printf("Enter the number of Monte Carlo trajectories (N) : ");
    scanf("%d", &N);
    if(argc > 1) N = atoi(argv[1]);

    // Constants
    float sqrtdt = sqrtf(dt);
    float C1 = 0.999f; 
    float C2 = 0.0001f;
    float C3 = 0.1f*sqrtdt*0.5f;
    float C4 = 0.1f*sqrtdt*sqrtf(0.75f);

    // Starting the timer right before entering the parallel part
    clock_t start_time, end_time;

    // Monte-Carlo CUDA kernel launch configuration
    const int BLOCK_SIZE_MC = 32; // Block size
    const int GRID_SIZE_MC = (N + BLOCK_SIZE_MC - 1) / BLOCK_SIZE_MC; // Grid size

    // Reduction CUDA kernel launch configuration
    const int BLOCK_SIZE_R = 512; // Block size
    const int GRID_SIZE_R = (N + BLOCK_SIZE_R - 1) / BLOCK_SIZE_R; // Grid size

    // Allocating memory for results on host
    float *h_results = (float *)malloc(N * sizeof(float));
    if (h_results == NULL) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }
    float* h_output = (float *)malloc(GRID_SIZE_R * sizeof(float));
    if (h_output == NULL) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Allocating memory for results on the device
    float *d_results;
    testCUDA(cudaMalloc((void **)&d_results, N * sizeof(float)));
    float* d_output;
    testCUDA(cudaMalloc(&d_output, GRID_SIZE_R * sizeof(float)));
    
    // Starting the timer right before entering the computing part
    start_time = clock();

    // Launch the Monte-Carlo CUDA kernel
    monte_carlo_kernel<<<GRID_SIZE_MC, BLOCK_SIZE_MC>>>(d_results, N, steps, sqrtdt, S0, Y0, K, C1, C2, C3, C4);

    // Launch reduction_kernel to reduce results
    reduction_kernel<<<GRID_SIZE_R, BLOCK_SIZE_R, BLOCK_SIZE_R * sizeof(float)>>>(d_results, d_output, N);
    cudaMemcpy(h_output, d_output, GRID_SIZE_R * sizeof(float), cudaMemcpyDeviceToHost);

    // Computing final option price by averaging
    float sum_payoff = 0.0f;
    for (int i = 0; i < GRID_SIZE_R; i++) {
        sum_payoff += h_output[i];
    }
    float option_price = sum_payoff / N;

    // Ending the timer and computing the time spent on the parallel part
    end_time = clock();

    // Freeing memory
    testCUDA(cudaFree(d_results));
    free(h_results);

    // Displaying the results
    // Should always be somewhere around 0.113
    printf("Call option price : %f\n", option_price);

    float execution_time = ((float)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Execution time of the Monte-Carlo loop : %f seconds\n", execution_time);

    return 0;
}
