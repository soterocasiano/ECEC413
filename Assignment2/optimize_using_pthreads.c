/* Implementation of PSO using pthreads.
 *
 * Author: Naga Kandasamy
 * Date: January 31, 2025
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "pso.h"
#include <pthread.h>


// Define a struct to pass data to each thread
typedef struct {
    int start;          // Start index for the thread
    int end;            // End index for the thread
    swarm_t *swarm;       // Pointer to the swarm
    double w, c1, c2;   // PSO parameters
    double xmin, xmax;  // Position bounds
    int iter;           // Current iteration
    char* function;     // function string
} ThreadData;


void *particle_update(void *arg) {
    ThreadData *data = (ThreadData *)arg; // cast arg to threaddata
    int i, j;
    particle_t *particle, *gbest; // init particles
    float r1, r2, curr_fitness;

    for (i = data->start; i < data->end; i++) {
        particle = &data->swarm->particle[i];
        gbest = &data->swarm->particle[particle->g];

        for (j = 0; j < particle->dim; j++) {
            r1 = (float)rand() / (float)RAND_MAX;
            r2 = (float)rand() / (float)RAND_MAX;

            // Update particle velocity
            particle->v[j] = data->w * particle->v[j] + data->c1 * r1 * (particle->pbest[j] - particle->x[j]) + data->c2 * r2 * (gbest->x[j] - particle->x[j]);

            // velocity
            if ((particle->v[j] < -fabsf(data->xmax - data->xmin)) || (particle->v[j] > fabsf(data->xmax - data->xmin))) {
                particle->v[j] = uniform(-fabsf(data->xmax - data->xmin), fabsf(data->xmax - data->xmin));
            }

            // Update particle position
            particle->x[j] = particle->x[j] + particle->v[j];

            if (particle->x[j] > data->xmax) particle->x[j] = data->xmax;
            if (particle->x[j] < data->xmin) particle->x[j] = data->xmin;
        }

        // Evaluate fitness
        pso_eval_fitness(data->function, particle, &curr_fitness);

        // Update pbest
        if (curr_fitness < particle->fitness) {
            particle->fitness = curr_fitness;
            for (j = 0; j < particle->dim; j++) {
                particle->pbest[j] = particle->x[j];
            }
        }
    }

    pthread_exit(NULL);
}


int solve_optimize_using_pthreads(char *function, int dim, swarm_t *swarm, 
                            float xmin, float xmax, int num_iter, int num_threads)
{
    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    ThreadData *thread_data = (ThreadData *)malloc(num_threads * sizeof(ThreadData));
    int iter = 0;
    int particles_per_thread = swarm->num_particles / num_threads;
    int i;

    while (iter < num_iter) {
        // Create threads
        for (i = 0; i < num_threads; i++) {
            thread_data[i].start = i * particles_per_thread;    // Particles per thread
            thread_data[i].end = (i == num_threads - 1) ? swarm->num_particles : (i + 1) * particles_per_thread; // Ternary operation to account for all particles in swarm 
            thread_data[i].swarm = swarm;
            thread_data[i].w = 0.79;        // Defined in assignment overview
            thread_data[i].c1 = 1.49;       // Defined in assignment overview
            thread_data[i].c2 = 1.49;       // Defined in assignment overview
            thread_data[i].xmin = xmin;     // Given by user
            thread_data[i].xmax = xmax;     // Given by user
            thread_data[i].iter = iter;     // Given by user
            thread_data[i].function = function;     // Maintain tracking of function

            pthread_create(&threads[i], NULL, particle_update, (void *)&thread_data[i]);    // Call the particle update function for each thread
        }

        // Join threads
        for (i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }

        iter++; // Increment iteration counter
    }

    free(threads);
    free(thread_data);
}

optimize_using_pthreads(char *function, int dim, int swarm_size, 
    float xmin, float xmax, int num_iter, int num_threads)
{
     /* Initialize PSO */
    swarm_t *swarm;
    srand(time(NULL));
    swarm = pso_init(function, dim, swarm_size, xmin, xmax); // Initialize the swarm
    if (swarm == NULL) {
        fprintf(stderr, "Unable to initialize PSO\n");
        exit(EXIT_FAILURE);
    }

#ifdef VERBOSE_DEBUG
    pso_print_swarm(swarm);
#endif

    /* Solve PSO */
    int g; 
    g = solve_optimize_using_pthreads(function, dim, swarm, xmax, xmin, num_iter, num_threads); // Call our pthreaded implementation
    if (g >= 0) {
        fprintf(stderr, "Solution:\n");
        pso_print_particle(&swarm->particle[g]); // Print the particle before returning
    }

    pso_free(swarm);
    return g;
}