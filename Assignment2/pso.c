/* Particle swarm optimizer.
 *
 * Note: This is an implementation of the original algorithm proposed in: 
 *
 * Yuhui Shi, "Particle Swarm Optimization," IEEE Neural Networks Society, pp. 8-13, February, 2004.
 *
 * Compile using provided Makefile: make 
 * If executable exists or if you have made changes to the .h file but not to the .c files, delete the executable and rebuild 
 * as follows: make clean && make
 *
 * Author: Naga Kandasamy
 * Date modified: January 31, 2025 
 *
 * Student/team: Sotero Casiano, Jeffrey Lau
 * Date: February 11, 2025
 */  
#include <stdio.h>
#include <stdlib.h>
#include "pso.h"
#include <sys/time.h>
#include <math.h>

int main(int argc, char **argv)
{
    if (argc < 8) {
        fprintf(stderr, "Usage: %s function-name dimension swarm-size xmin xmax max-iter num-threads\n", argv[0]);
        fprintf(stderr, "function-name: name of function to optimize\n");
        fprintf(stderr, "dimension: dimensionality of search space\n");
        fprintf(stderr, "swarm-size: number of particles in swarm\n");
        fprintf(stderr, "xmin, xmax: lower and upper bounds on search domain\n");
        fprintf(stderr, "max-iter: number of iterations to run the optimizer\n");
        fprintf(stderr, "num-threads: number of threads to create\n");
        exit(EXIT_FAILURE);
    }

    char *function = argv[1];
    int dim = atoi(argv[2]);
    int swarm_size = atoi(argv[3]);
    float xmin = atof(argv[4]);
    float xmax = atof(argv[5]);
    int max_iter = atoi(argv[6]);
    int num_threads = atoi(argv[7]);

    /* Optimize using reference version */
    int status;
    status = optimize_gold(function, dim, swarm_size, xmin, xmax, max_iter);
    if (status < 0) {
        fprintf(stderr, "Error optimizing function using reference code\n");
        exit (EXIT_FAILURE);
    }

    struct timeval start, stop;	
	gettimeofday(&start, NULL);
    /* FIXME: Complete this function to perform PSO using pthreads. 
     * Return -1 on error, 0 on success. Print best-performing 
     * particle within the function prior to returning. 
     */
    status = optimize_using_pthreads(function, dim, swarm_size, xmin, xmax, max_iter, num_threads);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
        + (stop.tv_usec - start.tv_usec)/(float)1000000));
        
    if (status < 0) {
        fprintf(stderr, "Error optimizing function using pthreads\n");
        exit (EXIT_FAILURE);
    }
    
    exit(EXIT_SUCCESS);
}

/* Print command-line arguments */
void print_args(char *function, int dim, int swarm_size, float xmin, float xmax)
{
    fprintf(stderr, "Function to optimize: %s\n", function);
    fprintf(stderr, "Dimensionality of search space: %d\n", dim);
    fprintf(stderr, "Number of particles: %d\n", swarm_size);
    fprintf(stderr, "xmin: %f\n", xmin);
    fprintf(stderr, "xmax: %f\n", xmax);
    return;
}

