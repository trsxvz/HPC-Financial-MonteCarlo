#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

// Main function
int main() {

    // Initialize the GSL random number generator
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, (unsigned long)time(NULL));
    
    // Model parameters
    float S0 = 1.0f, Y0 = logf(0.1f);
    float K = 1.0f, T = 1.0f;
    float dt = 0.001f ;
    int steps = (int)(T/dt) ;
    // Precomputing the constants
    float sqrtdt = sqrtf(dt);
    float C1 = 0.999f; 
    float C2 = 0.0001f;
    float C3 = 0.1f*sqrtdt*0.5f;
    float C4 = 0.1f*sqrtdt*sqrtf(0.75f);

    // Initial conditions
    float S, Y, payoff, sum_payoff = 0.0f;

    // Asking the user how much Monte-Carlo trajectories he wants to run
    int N;
    printf("Enter the number of Monte Carlo trajectories (N) : ");
    scanf("%d", &N);

    // Starting the timer right before entering the computing part
    clock_t start_time, end_time;
    start_time = clock();

    // Monte-Carlo loop with N different simulations (aka trajectories)
    for (int i = 0; i < N; i++) {

        // Initial conditions
        S = S0;
        Y = Y0;

        // Simulation on the Euler scheme for j from 0 to last_step
        for (int j = 0; j < steps; j++) {
            // Generation of two normal standard random variables with GSL's Ziggurat
            float G1 = gsl_ran_gaussian_ziggurat(rng, 1.0); 
            float G2 = gsl_ran_gaussian_ziggurat(rng, 1.0);
            // Updating St and Yt
            S = S * (1.0f + expf(Y) * sqrtdt * G1);
            Y = C1 * Y + C2 - C3 * G1 + C4 * G2 ;
        }

        // Evaluating the payoff of the finished trajectory
        payoff = fmax(S - K, 0.0f);

        // Updating the sum of simulated payoffs
        sum_payoff += payoff;
    }

    // Computing final option price by averaging
    float option_price = sum_payoff / N;

    // Ending the timer and evaluating time spent in the computing part
    end_time = clock();
    float execution_time = ((float)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Free the GSL random number generator
    gsl_rng_free(rng);

    // Displaying the results
    printf("Call option price : %f\n", option_price);
    printf("Computing time : %f seconds\n", execution_time);

    return 0;
}