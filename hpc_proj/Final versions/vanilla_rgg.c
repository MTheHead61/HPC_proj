#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>

#define NUM_RUNS 1000

#define DIM 1000

#define R 1.0

#define SEED 349872935

int main(int argc, char *argv[]) {
    int i, j;
    double r, x, y;
    double dist = R;
    unsigned int N = DIM; // Number of nodes
    double* X = (double*)malloc(N*sizeof(double*)); // Array with the nodes abscissa
    double* Y = (double*)malloc(N*sizeof(double*)); // Array with the nodes ordinate
    int* graph = (int*)malloc(N*N*sizeof(int*));
    clock_t start, end;
    double cpu_time_used;

    // Set the random seed
    srand(SEED);

    // Start timing
    start = clock();

    // Create random N points in a 2-D space
    for (i=0; i<N; i++){
        X[i] = (double)rand()/RAND_MAX; // We get a random number in interval [0,1[
        Y[i] = (double)rand()/RAND_MAX;
    }

    for (int run=0; run<NUM_RUNS; run++) {
        for (i=0; i<N*N; i++) { graph[i] = 0; } // Initialize the matrix

        for (i=0; i<N; i++){

            for (j=i+1; j<N; j++){
                // We prefer to use two more variables (x and y), instead of doing just
                //one operation ( r = sqrt(pow((X[i]-X[j]), 2) + pow((Y[i]-Y[j]), 2)) ) because this is less
                //computationally heavy
                x = X[i]-X[j]; // Distance along the x axis
                y = Y[i]-Y[j]; // Distance along the y axis
                r = x*x + y*y;
                r = sqrt(r);

                if (r<dist){ // If the distance between the two nodes is less than R then we will connect them
                    graph[i*N+j] = 1;
                    graph[j*N+i] = 1;
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
