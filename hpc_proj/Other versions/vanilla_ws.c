#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_RUNS 10

#define DIM 100

#define P 0.3

#define INIT_CONN 4

#define SEED 349872935

#define BASE 569872384

#define TS 10 // Threadblock sizes

int main() {
    unsigned int N = DIM;          // Number of nodes
    double PROB = P;    // Probability of edge creation
    int K = INIT_CONN;
    double r;
    int i, j, l;
    int* G = (int*)malloc(N*N*sizeof(int*));
    clock_t start, end;
    double cpu_time_used;

    // Set the random seed
    srand(SEED);

    // Start timing
    start = clock();

    // Initialize the adjacency matrix
    for (int run=0; run<NUM_RUNS; run++) {
        for (i=0; i<N*N; i++) { G[i] = 0; }
        for (i = 0; i < N; i++) {
            for (j = 1; j <= K/2; j++){
                r = (double)rand() / RAND_MAX;
                if (r<PROB) {
                    do{
                        l = (rand() % (N-K-2)) + (K/2) + 1;
                    } while (G[i*N+((i+l)%N)] == 1);
                    G[i*N+((i+l)%N)] = 1;
                    G[((i+l)%N)*N+i] = 1;
                }
                else {
                    G[i*N+((i+j)%N)] = 1;
                    G[((i+j)%N)*N+i] = 1;
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
            printf("%d ", G[i*N+j]);
        }
        printf("\n");
    }*/

    printf("Mean time taken to create the graph: %f seconds\n", cpu_time_used);

    return 0;
}
