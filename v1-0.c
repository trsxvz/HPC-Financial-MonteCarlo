#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define M_PI 3.1415927f

// Box-Muller method for generating standard normal random variables
float generate_normal() {

    // Generate two uniform random variables in [0,1]
    float u1, u2;
    do {
    u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    } while (u1 == 0.0f || u2 == 0.0f);

    // Apply the Box-Muller transform
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return z0;
}

// Main function
int main() {

    // Time-based seed for the random generator
    srand(time(NULL));
    
    // Model parameters
    float alpha = 1.0f, m = 0.1f, beta = 0.1f, rho = -0.5f;
    float S0 = 1.0f, Y0 = logf(0.1f);
    float K = 1.0f, T = 1.0f;
    float dt = 0.001f ;
    int steps = (int)(T/dt) ;
    // Simulation variables
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
            // Generation of two normal standard random variables with Box-Muller
            float G1 = generate_normal();
            float G2 = generate_normal();
            // Updating St and Yt
            S = S + expf(Y) * S * sqrtf(dt) * G1;
            Y = Y + alpha * (m - Y) * dt + beta * sqrtf(dt) * (rho * G1 + sqrtf(1.0f - rho * rho) * G2);
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

    // Displaying the results
    printf("Call option price : %f\n", option_price);
    printf("Computing time : %f seconds\n", execution_time);

    return 0;
}