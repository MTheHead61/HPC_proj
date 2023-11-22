#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SEED 12345  // Fixed random seed

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <number_of_nodes> <probability>\n", argv[0]);
        return 1;
    }

    unsigned int N = atoi(argv[1]);          // Number of nodes
    double PROB = atof(argv[2]);    // Probability of edge creation
    int i, j;
    int graph[N][N];
    clock_t start, end;
    double cpu_time_used;

    // Set the random seed
    srand(SEED);

    // Start timing
    start = clock();

    // Initialize the adjacency matrix
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i == j) {
                graph[i][j] = 0;  // No self-loops
            } else {
                // Generate a random number between 0 and 1
                double r = (double)rand() / RAND_MAX;
                // Check if an edge should be created
                graph[i][j] = (r < PROB) ? 1 : 0;
                graph[j][i] = graph[i][j]; // Undirected graph, so mirror the edge
            }
        }
    }

    // End timing
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Print the adjacency matrix
   /* printf("Adjacency Matrix:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%d ", graph[i][j]);
        }
        printf("\n");
    }*/

    printf("Time taken to create the graph: %f seconds\n", cpu_time_used);

    return 0;
}
