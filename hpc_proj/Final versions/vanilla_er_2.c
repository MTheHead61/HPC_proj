#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#define NUM_RUNS 1000

#define DIM 10000

#define P 0.0

#define SEED 349872935

int main(int argc, char *argv[]) {
    unsigned int N = DIM; // Number of nodes
    double p = P; // Probability of edge creation
    double r;
    int i, j;
    int* graph = (int*)malloc(N*N*sizeof(int*));
    clock_t start, end;
    double cpu_time_used;

    // Set the random seed
    srand(SEED);

    // Start timing
    start = clock();

    // Initialize the adjacency matrix
    for (int run=0; run<NUM_RUNS; run++) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                if (i == j) {
                    graph[i*N+j] = 0;  // No self-loops
                }
                else {
                    // Generate a random number between 0 and 1
                    r = (double)rand() / RAND_MAX;
                    // Check if an edge should be created
                    graph[i*N+j] = (r < p) ? 1 : 0;
                }
            }
        }
    }
    // End timing
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    cpu_time_used /= NUM_RUNS;

    // Print the adjacency matrix
    /*printf("Adjacency Matrix:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%d ", graph[i*N+j]);
        }
        printf("\n");
    }*/
    int Digs = DECIMAL_DIG;
    printf("Mean time taken to create the graph: %.*e seconds\n", Digs, cpu_time_used);

    return 0;
}
