#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_RUNS 1000

#define DIM 10

#define M 2

#define SEED 349872935

#define BASE 569872384

#define TS 10 // Threadblock sizes

int main(int argc, char *argv[]) {
    int i, j, k, m, a, r;
    unsigned int N = DIM; // Number of nodes
    int* deg = (int*)malloc(N*sizeof(int*)); // Array with the degrees of the nodes
    int* graph = (int*)malloc(N*N*sizeof(int*));
    unsigned int edges; // We just need the number of edges because we know that the sum of the degrees
    //in an undirected graph is 2|E|
    clock_t start, end;
    double cpu_time_used;

    // Set the random seed
    srand(SEED);

    // Start timing
    start = clock();

    // Initialize the adjacency matrix
    for (int run=0; run<NUM_RUNS; run++) {
        for (i=0; i<N; i++) { deg[i] = 0; } // Initialize the degree array
        for (i=0; i<N*N; i++) { graph[i] = 0; } // Initialize the matrix
        edges = 0;
        
        // Set the starting M+1 nodes with M starting edges
        for (i = 1; i <= M; i++){
            graph[i] = 1;
            graph[i*N] = 1;
            deg[i] += 1;
            edges += 1;
        }
        deg[0] = M;

        for (i = M+1; i < N; i++) { // For every node
            m = 0;
            while (deg[i] < M) { // For M times
                r = rand()%(edges*2); // We find our random value in a natural range of numbers instead of
                //[0,1]. This way we will need much less operations
                for (j = 0; j<N; j++) { // For every node we can connect to
                    a = 0; // Initialize our probability
                    for (k = 0; k<=j; k++) {
                        a += deg[k]; //Compute the probability
                    }
                    if (r < a) {
                        if (graph[i*N+j] == 0 ){
                            graph[i*N+j] = 1;
                            graph[j*N+i] = 1;
                            deg[i] += 1;
                            deg[j] += 1;
                            edges += 1;
                            m += 1;
                            break;
                        }
                    }
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

    printf("Mean time taken to create the graph: %f seconds\n", cpu_time_used);

    return 0;
}
