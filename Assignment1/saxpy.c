/* Implementation of the SAXPY loop.
 *
 * Compile as follows: gcc -o saxpy saxpy.c -O3 -Wall -std=c99 -pthread -lm
 *
 * Author: Naga Kandasamy
 * Date modified: January 23, 2025 
 *
 * Student names: Sotero Casiano, Jeffrey Lau
 * Date: 1/28/2025
 *
 * */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>

/* Structure used to pass arguments to the worker threads*/
typedef struct args_for_thread {
    int tid;                /* Thread ID */
    float* x;               /* x */
    float* y;             /* y */
    float a;                /* Scalar value*/
    int chunk_size;             /* num_elements % num_threads */
    int num_threads;         /* Number of threads*/
    int num_elements;       /* Number of elements*/
} args_for_thread_t;

/* Function prototypes */
void compute_gold(float *, float *, float, int);
void compute_using_pthreads_v1(float *, float *, float, int, int);
void compute_using_pthreads_v2(float *, float *, float, int, int);
int check_results(float *, float *, int, float);

int main(int argc, char **argv)
{
	if (argc < 3) {
		fprintf(stderr, "Usage: %s num-elements num-threads\n", argv[0]);
        fprintf(stderr, "num-elements: Number of elements in the input vectors\n");
        fprintf(stderr, "num-threads: Number of threads\n");
		exit(EXIT_FAILURE);
	}
	
    int num_elements = atoi(argv[1]); 
    int num_threads = atoi(argv[2]);

	/* Create vectors X and Y and fill them with random numbers between [-.5, .5] */
    fprintf(stderr, "Generating input vectors\n");
    int i;
	float *x = (float *)malloc(sizeof(float) * num_elements);
    float *y1 = (float *)malloc(sizeof(float) * num_elements);              /* For the reference version */
	float *y2 = (float *)malloc(sizeof(float) * num_elements);              /* For pthreads version 1 */
	float *y3 = (float *)malloc(sizeof(float) * num_elements);              /* For pthreads version 2 */

	srand(time(NULL)); /* Seed random number generator */
	for (i = 0; i < num_elements; i++) {
		x[i] = rand()/(float)RAND_MAX - 0.5;
		y1[i] = rand()/(float)RAND_MAX - 0.5;
        y2[i] = y1[i]; /* Make copies of y1 for y2 and y3 */
        y3[i] = y1[i]; 
	}

    float a = 2.5;  /* Choose some scalar value for a */

	/* Calculate SAXPY using the reference solution. The resulting values are placed in y1 */
    fprintf(stderr, "\nCalculating SAXPY using reference solution\n");
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	
    compute_gold(x, y1, a, num_elements); 
	
    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

	/* Compute SAXPY using pthreads, version 1. Results must be placed in y2 */
    fprintf(stderr, "\nCalculating SAXPY using pthreads, version 1\n");
	gettimeofday(&start, NULL);

	compute_using_pthreads_v1(x, y2, a, num_elements, num_threads);
	
    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Compute SAXPY using pthreads, version 2. Results must be placed in y3 */
    fprintf(stderr, "\nCalculating SAXPY using pthreads, version 2\n");
	gettimeofday(&start, NULL);

	compute_using_pthreads_v2(x, y3, a, num_elements, num_threads);
	
    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Check results for correctness */
    fprintf(stderr, "\nChecking results for correctness\n");
    float eps = 1e-12;                                      /* Do not change this value */
    if (check_results(y1, y2, num_elements, eps) == 0)
        fprintf(stderr, "TEST PASSED\n");
    else 
        fprintf(stderr, "TEST FAILED\n");
 
    if (check_results(y1, y3, num_elements, eps) == 0)
        fprintf(stderr, "TEST PASSED\n");
    else 
        fprintf(stderr, "TEST FAILED\n");

	/* Free memory */ 
	free((void *)x);
	free((void *)y1);
    free((void *)y2);
	free((void *)y3);

    exit(EXIT_SUCCESS);
}

/* Compute reference soution using a single thread */
void compute_gold(float *x, float *y, float a, int num_elements)
{
	int i;
    for (i = 0; i < num_elements; i++)
        y[i] = a * x[i] + y[i]; 
}

void *compute_gold_with_chunking(void *args)
{
    args_for_thread_t *thread_data = (args_for_thread_t *)args;
	int i;

    if (thread_data->tid < (thread_data->num_threads - 1)){
        for (i = thread_data->tid * thread_data->chunk_size; i < (thread_data->tid + 1) * thread_data->chunk_size; i++)
            thread_data->y[i] = thread_data->a * thread_data->x[i] + thread_data->y[i];
    }
    else{
        for (i = thread_data->tid * thread_data->chunk_size; i < thread_data->num_elements; i++)
            thread_data->y[i] = thread_data->a * thread_data->x[i] + thread_data->y[i];
    }

    /* Free data structures */
    free((void *)thread_data);
    pthread_exit(NULL);
}

/* Function prototype for the thread routines */
//void *worker(void *);


/* Calculate SAXPY using pthreads, version 1. Place result in the Y vector */
void compute_using_pthreads_v1(float *x, float *y, float a, int num_elements, int num_threads)
{
    /* Allocate memory to store the IDs of the worker threads */
    pthread_t *worker_thread = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    args_for_thread_t *args_for_thread;
	
    int i;
	int diff = floor(num_elements / num_threads);
    /* Fork point: create worker threads and ask them to execute worker that takes a structure as an argument */
    for (i = 0; i < num_threads; i++) {
        args_for_thread = (args_for_thread_t *)malloc(sizeof(args_for_thread_t)); /* Memory for structure to pack the arguments */
        args_for_thread->tid = i; /* Fill the structure with some dummy arguments */
        args_for_thread->x = x; 
        args_for_thread->y = y;
        args_for_thread->a = a;
        args_for_thread->chunk_size = diff; 
        args_for_thread->num_threads = num_threads;
        args_for_thread->num_elements = num_elements;
		
        if ((pthread_create(&worker_thread[i], NULL, compute_gold_with_chunking, (void *)args_for_thread)) != 0) {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }
    }
		  
    /* Join point: wait for all the worker threads to finish */
    for (i = 0; i < num_threads; i++)
        pthread_join(worker_thread[i], NULL);
		
}

void *compute_gold_with_striding(void *args)
{
    args_for_thread_t *thread_data = (args_for_thread_t *)args;
    int tid = thread_data->tid;

    while (tid < thread_data->num_elements){
        thread_data->y[tid] = thread_data->a * thread_data->x[tid] + thread_data->y[tid];
        tid = tid + thread_data->num_threads;
    }

    /* Free data structures */
    free((void *)thread_data);
    pthread_exit(NULL);
}

/* Calculate SAXPY using pthreads, version 2. Place result in the Y vector */
void compute_using_pthreads_v2(float *x, float *y, float a, int num_elements, int num_threads)
{
    /* Allocate memory to store the IDs of the worker threads */
    pthread_t *worker_thread = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    args_for_thread_t *args_for_thread;
	
    int i;

    /* Fork point: create worker threads and ask them to execute worker that takes a structure as an argument */
    for (i = 0; i < num_threads; i++) {
        args_for_thread = (args_for_thread_t *)malloc(sizeof(args_for_thread_t)); /* Memory for structure to pack the arguments */
        args_for_thread->tid = i; /* Fill the structure with some dummy arguments */
        args_for_thread->x = x; 
        args_for_thread->y = y;
        args_for_thread->a = a;
        args_for_thread->chunk_size = 1; 
        args_for_thread->num_threads = num_threads;
        args_for_thread->num_elements = num_elements;
		
        if ((pthread_create(&worker_thread[i], NULL, compute_gold_with_striding, (void *)args_for_thread)) != 0) {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }
    }
		  
    /* Join point: wait for all the worker threads to finish */
    for (i = 0; i < num_threads; i++)
        pthread_join(worker_thread[i], NULL);
}

/* Perform element-by-element check of vector if relative error is within specified threshold */
int check_results(float *A, float *B, int num_elements, float threshold)
{
    int i;
    for (i = 0; i < num_elements; i++) {
        if (fabsf((A[i] - B[i])/A[i]) > threshold)
            return -1;
    }
    
    return 0;
}



