/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>



/* ---------- BASIC CUDA FUNCTIONS ---------- */



// Device arrays to store the exploration ranges for Y0, m, alpha, nu2, and rho
// These arrays represent the parameter space for the simulation
__device__ float Y0d[10];
__device__ float md[10];
__device__ float alphad[10];
__device__ float nu2d[10];
__device__ float rhod[10];
__device__ float Kd[16];

// Function that catches the CUDA error 
void testCUDA(cudaError_t error, const char* file, int line) {
	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d : %s\n", file, line, cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

// Function to set the random state for each thread
__global__ void init_curand_state_k(curandState* state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init((unsigned long long)clock64() + idx, idx, 0, &state[idx]);
}



/* ---------- CUDA MONTE-CARLO KERNEL ---------- */



// Monte Carlo simulation kernel
__global__ void MC_k(float dt, float sqrtdt, int steps, float T, int N, curandState* state, float* results){

	// Determining global thread ID and assigning a random state
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	curandState localState = state[idx];

	// Declaring local variables
	float S, Y, price, G1, G2; float2 randPair;

	// Initializing local varying parameters
	float KR = Kd[idx / 100000];
	float Y0R = Y0d[(idx % 100000) / 10000];
	float mR = md[(idx % 10000) / 1000];
	float rhoR = rhod[(idx % 1000) / 100];
	float alphaR = alphad[(idx % 100) / 10];
	float nu2R = nu2d[idx % 10];

	// Precomputing constants
	float betaR = sqrtf(2.0f*alphaR*nu2R) * (1.0f - expf(mR)) ;
	float C1 = (1.0f - dt * alphaR) ;
	float C2 = dt * alphaR * mR ;
	float C3 = betaR * sqrtdt * rhoR ;
	float C4 = betaR * sqrtdt * sqrtf(1.0f - (rhoR * rhoR)) ;
	
	// Declaring a local accumulator for the Monte-Carlo results
	float sumPayoffs = 0.0f, sumSquaredPayoffs = 0.0f;

	for (int i = 0; i < N; i++) {

		// Initial conditions
		S = 1.0f;
		Y = Y0R;

		// Computing for every time step
		for (int j = 0; j < steps ; j++) {

			// Updating values
			randPair = curand_normal2(&localState);
			G1 = randPair.x;
            G2 = randPair.y;
            S = S * (1.0f + expf(Y) * sqrtdt * G1);
            Y = Y * C1 + C2 + C3 * G1 + C4 * G2 ;

		}

		// Avoiding extreme values of S distorting the upcoming NN
		if (S < 12.0f) {
			price = S-KR > 0.0f ? S-KR: 0.0f;
			sumPayoffs += price;
			sumSquaredPayoffs += price * price;
		}
	}

	// Copying the results and square results to the results[] array
	results[2 * idx] = sumPayoffs / N;
	results[2 * idx + 1] = sumSquaredPayoffs / N;

	// Copying random state back to global memory
	state[idx] = localState;
}



/* ---------- MISC HOST FUNCTIONS ---------- */



// Function to compute K the (strike) depending on T
void strikeInterval(float* K, float T) {

	float fidx = T * 12.0f + 1.0f;
	int i = 0;
	float coef = 1.0f;
	float delta;

	while (i < fidx) {
		coef *= (1.02f);
		i++;
	}

	delta = pow(coef, 1.0f / 8.0f);
	K[15] = coef;

	for (i = 1; i < 16; i++) {
		K[15 - i] = K[15 - i + 1] / delta;
	}
}



/* ---------- MAIN FUNCTION ---------- */



int main(void) {

	// Defining the number of MC trajectories (N) and the size of the time steps
	int N = 256*512;
	float dt = 1.0f/1000.0f ;
	float sqrtdt = sqrtf(dt);
	int steps;

	// Exploring 10 decreasing "Y0" values from log(0.08) to log(0.4)
	float Y0[10] = {logf(0.4f), logf(0.35f), logf(0.31f), logf(0.27f), logf(0.23f), 
					logf(0.2f), logf(0.17f), logf(0.14f), logf(0.11f), logf(0.08f)};

	// Exploring 10 decreasing "m" values from log(0.06) to log(0.34)
	float m[10] = {logf(0.34f), logf(0.3f), logf(0.27f), logf(0.24f), logf(0.21f), 
					logf(0.18f), logf(0.15f), logf(0.12f), logf(0.09f), logf(0.06f)};

	// Exploring 10 increasing "alpha" values from 0.1 to 51.2
	float alpha[10] = { 0.1f, 0.2f, 0.4f, 0.8f, 1.6f, 3.2f, 6.4f, 12.8f, 25.6f, 51.2f };

	// Exploring 10 increasing "nu2" values from 0.6 to 1.5
	float nu2[10] = { 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f };

	// Exploring 10 decreasing "rho" values from -0.95 to 0.95
	float rho[10] = {0.95f, 0.75f, 0.55f, 0.35f, 0.15f, -0.15f, -0.35f, -0.55f, -0.75f, -0.95f};
	
	// Exploring 16 increasing "T" values from 1/12 to 2
	float T[16] = { 1.0f / 12.0f,  2.0f / 12.0f, 3.0f / 12.0f, 4.0f / 12.0f, 5.0f / 12.0f, 6.0f / 12.0f, 7.0f / 12.0f,
					  8.0f / 12.0f, 9.0f / 12.0f, 10.0f / 12.0f, 11.0f / 12.0f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f };

	// Declaring the 16 "K" values array.
	// It will be defined dynamically depending on T.
	float K[16];

	// Copying some of these exploration ranges to the GPU device
	cudaMemcpyToSymbol(Y0d, Y0, 10*sizeof(float));
	cudaMemcpyToSymbol(md, m, 10*sizeof(float));
	cudaMemcpyToSymbol(alphad, alpha, 10*sizeof(float));
	cudaMemcpyToSymbol(nu2d, nu2, 10*sizeof(float));
	cudaMemcpyToSymbol(rhod, rho, 10*sizeof(float));

	// CUDA launch configuration
	// 128*4*625*5 = 1.6 million threads.
	// Each MC kernel call goes through 16 values for K and all parameters combinations (100k) 
	// 16*100k = 1.6mil threads so this fits perfectly.
	int BLOCK_SIZE = 128 * 4;
	int GRID_SIZE = 625 * 5;

	// Declaring variables on the host (R stands for Replica)
	float KR, mR, alphaR, nu2R, rhoR, Y0R, price, error;

	// Declaring, allocating and generating random CUDA states
	curandState* states;
	cudaMalloc(&states, GRID_SIZE*BLOCK_SIZE*sizeof(curandState));
	init_curand_state_k <<<GRID_SIZE, BLOCK_SIZE>>> (states);

	// Declaring a pointer for the MC results array
	float *mcResults;

	// Declaring a file pointer and a string buffer to construct file names
	FILE* fpt;
	char strg[30];

	// Declaring the "same" index. It will be used to locate the results of a specific 
	// parameters combination in the MC results array, which is called mcResults[]
	int same;

	// Starting the timer right before entering the computing part
	printf("\nComputing Monte-Carlo simulations. Please wait...\n");
	clock_t start_time_outer, end_time_outer, start_time_mc, end_time_mc;
    start_time_outer = clock();

	// Looping through all 16 possible values of maturity (T)
	for(int i=0; i<16; i++){

		// Allocating memory (doing this in the loop divides its size by 16)
		// If done outside the loop, this uses 175 additional MB of gpu memory
		cudaMallocManaged(&mcResults, 2*GRID_SIZE*BLOCK_SIZE*sizeof(float));

		// Computing the values of the K array depending on T[i]
		// Then copying it into the GPU device memory
		strikeInterval(K, T[i]);
		cudaMemcpyToSymbol(Kd, K, 16*sizeof(float));

		// Launching the MC kernel for one of the 16 values of T
		// This kernel is launched 16 times (because of the above for loop on i)
		steps = int(T[i]/dt);
		start_time_mc = clock();
		MC_k<<<GRID_SIZE,BLOCK_SIZE>>>(dt, sqrtdt, steps, T[i], N, states, mcResults);
		cudaDeviceSynchronize();
		end_time_mc = clock();
		int execution_time_mc = ((int)(end_time_mc - start_time_mc)) / CLOCKS_PER_SEC;
		printf("\n-------------------------------------------------------------------");
		printf("\n%.2f percent simulations (%d/16 MCkernel calls) have been computed. ", ((i+1)*100.0f/16), (i+1));
		printf("\n%d seconds have elapsed since the last MCkernel call completed.", execution_time_mc);
		printf("\n-------------------------------------------------------------------");

		// LOOPING THROUGH THE RESULTS ON THE HOST NOW
		// This loop just writes the results in a file

		// Looping through all 16 possible values of strike (K)
		for(int j=0; j<16; j++){

			// Starting the file writing timer
			start_time_mc = clock();

			// Creating a file for the current T[i] and K[j] values
			KR = K[j];
			sprintf(strg, "T%.4fK%.4f.csv", T[i], KR);

			// Opening the file in write mode and writing its header (column values)
			fpt = fopen(strg, "w+");
			fprintf(fpt, "alpha, nu2, m, rho, Y0, price, 95cI\n");

			// Looping through all 100k (=3125*32) other parameters combinations
			for(int o=0; o < 3125*32; o++){

				// Updating the "same" index to locate the result obtained with the
				// current parameters combination in the "results[]" array
				same = o + j*(3125*32);

				// Finding the price of the current parameters combination thanks to the "same" index
				price = mcResults[2*same];

				// Determining the confidence interval
				error = 1.96f*sqrtf(mcResults[2*same+1] - (price * price)) / sqrtf((float)N);
                
				// Determining corresponding values of the parameters
    			Y0R = Y0[(same % 100000) / 10000];
                mR = m[(same % 10000) / 1000];
                rhoR = rho[(same % 1000) / 100];
                alphaR = alpha[(same % 100) / 10];
                nu2R = nu2[same % 10];
				
				// Writing the results into the current opened file
				fprintf(fpt, "%f, %f, %f, %f, %f, %f, %f\n", alphaR, nu2R, mR, rhoR, Y0R, price, error);
			}

			// Closing the file
			fclose(fpt);

			// Displaying the time spent writing the file
			end_time_mc = clock();
			execution_time_mc = ((int)(end_time_mc - start_time_mc)) / CLOCKS_PER_SEC;
			printf("\nFile %d/256 has been written in %d seconds", ((i*16)+j+1), execution_time_mc);

		}

		cudaFree(mcResults);

	}

	// Ending the timer and evaluating time spent in the outer (mc+filewriting) computing part
    end_time_outer = clock();
    float execution_time_outer = ((float)(end_time_outer - start_time_outer)) / CLOCKS_PER_SEC;

	// Freeing memory
	cudaFree(states);

	// Displaying the execution time results
	printf("\n\n-------------------------------------------------------------------");
	printf("\nEXECUTION COMPLETE !");
    printf("\nGlobal computing time of the program : %f seconds\n", execution_time_outer); 
	printf("\n-------------------------------------------------------------------");

	return 0;
}