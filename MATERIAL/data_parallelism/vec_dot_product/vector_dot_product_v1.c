/* Program calculates the vector dot product of two vectors A and B, P = A.B, using pthreads. 
 * This version does not make use of any atomics.
 *  
 * Compile as follows: gcc -o vector_dot_product_v1 vector_dot_product_v1.c -O3 -Wall -std=c99 -lpthread -lm
 *
 * Author: Naga Kandasamy
 * Date created: April 4, 2011
 * Date modified: April 9, 2020 
 * */

#define _REENTRANT /* Make sure the library functions are MT (muti-thread) safe */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>

/* Data structure defining what to pass to each worker thread */
typedef struct thread_data_s {
    int tid;                        /* The thread ID */
    int num_threads;                /* Number of threads in the worker pool */
    int num_elements;               /* Number of elements in the vector */
    float *vector_a;                /* Pointer to vector_a */
    float *vector_b;                /* Pointer to vector_b */
    int offset;                     /* Starting offset for each thread within the vectors */ 
    int chunk_size;                 /* Chunk size */
    double *partial_sum;            /* Starting address of the partial_sum array */
} thread_data_t;


/* Function prototypes */
double compute_gold(float *, float *, int);
double compute_using_pthreads(float *, float *, int, int);
void *dot_product(void *);
void print_args(thread_data_t *);


int main(int argc, char **argv)
{
	if (argc < 3) {
		printf("Usage: %s num-elements num-threads\n", argv[0]);
        printf("num-elements: The number of elements in the input vectors\n");
        printf("num-threads: The number of threads\n");
		exit(EXIT_FAILURE);
	}
	
    int num_elements = atoi(argv[1]); 
    int num_threads = atoi(argv[2]);

	/* Create the vectors A and B and fill them with random numbers between [-.5, .5] */
	float *vector_a = (float *)malloc(sizeof(float) * num_elements);
	float *vector_b = (float *)malloc(sizeof(float) * num_elements); 
	srand(time(NULL)); /* Seed random number generator */
	for (int i = 0; i < num_elements; i++) {
		vector_a[i] = rand()/(float)RAND_MAX - 0.5;
		vector_b[i] = rand()/(float)RAND_MAX - 0.5;
	}

	/* Compute dot product using the reference solution */
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	
    double reference = compute_gold(vector_a, vector_b, num_elements); 
	
    gettimeofday(&stop, NULL);
	printf("Reference solution = %f\n", reference);
	printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));
	printf("\n");

	/* Compute the dot product using the multi-threaded solution */
	gettimeofday(&start, NULL);

	double mt_result = compute_using_pthreads(vector_a, vector_b, num_elements, num_threads);
	
    gettimeofday(&stop, NULL);
	printf("Pthread solution = %f\n", mt_result);
	printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));
	printf("\n");

	/* Free memory here */ 
	free((void *)vector_a);
	free((void *)vector_b);

	pthread_exit(NULL);
}

/* Compute reference soution using a single thread */
double compute_gold(float *vector_a, float *vector_b, int num_elements)
{
	double sum = 0.0;
	for (int i = 0; i < num_elements; i++)
        sum += vector_a[i] * vector_b[i];
	
	return sum;
}

/* This function computes the dot product using pthreads */
double compute_using_pthreads(float *vector_a, float *vector_b, int num_elements, int num_threads)
{
    pthread_t *thread_id = (pthread_t *)malloc (num_threads * sizeof(pthread_t)); /* Data structure to store the thread IDs */
    pthread_attr_t attributes;      /* Thread attributes */
    pthread_attr_init(&attributes); /* Initialize thread attributes to default values */
		  
    /* Fork point: allocate memory on heap for required data structures and create worker threads */
    int i;
    int chunk_size = (int)floor((float)num_elements/(float) num_threads); /* Compute the chunk size */
    double *partial_sum = (double *)malloc(sizeof(double) * num_threads); /* Allocate memory to store partial sums returned by threads */
    
    thread_data_t *thread_data = (thread_data_t *) malloc(sizeof(thread_data_t) * num_threads);	  
    for (i = 0; i < num_threads; i++) {
        thread_data[i].tid = i; 
        thread_data[i].num_threads = num_threads;
        thread_data[i].num_elements = num_elements; 
        thread_data[i].vector_a = vector_a; 
        thread_data[i].vector_b = vector_b; 
        thread_data[i].offset = i * chunk_size; 
        thread_data[i].chunk_size = chunk_size;
        thread_data[i].partial_sum = partial_sum; 
    }

    for (i = 0; i < num_threads; i++)
        pthread_create(&thread_id[i], &attributes, dot_product, (void *)&thread_data[i]);
					 
    /* Join point: wait for the workers to finish */
    for (i = 0; i < num_threads; i++)
        pthread_join(thread_id[i], NULL);
		 
    /* Accumulate partial sums */ 
    double sum = 0.0; 
    for (i = 0; i < num_threads; i++)
        sum += partial_sum[i];
		
    /* Free dynamically allocated data structures */
    free((void *)partial_sum);
    free((void *)thread_data);

    return sum;
}

/* Function executed by each thread to compute the overall dot product */
void *dot_product(void *args)
{
    /* Typecast argument as a pointer to the thread_data_t structure */
    thread_data_t *thread_data = (thread_data_t *)args; 		  
	
    /* Compute the partial sum that this thread is responsible for. */
    double partial_sum = 0.0; 
    if (thread_data->tid < (thread_data->num_threads - 1)) {
        for (int i = thread_data->offset; i < (thread_data->offset + thread_data->chunk_size); i++)
            partial_sum += thread_data->vector_a[i] * thread_data->vector_b[i];
    } 
    else { /* This takes care of the number of elements that the final thread must process */
        for (int i = thread_data->offset; i < thread_data->num_elements; i++)
            partial_sum += thread_data->vector_a[i] * thread_data->vector_b[i];
    }
		  
    /* Store partial sum into the partial_sum array */
    thread_data->partial_sum[thread_data->tid] = partial_sum;
		  
    pthread_exit(NULL);
}

/* Helper function */
void print_args(thread_data_t *thread_data)
{
    printf("Thread ID: %d\n", thread_data->tid);
    printf("Numer of threads: %d\n", thread_data->num_threads);
    printf("Num elements: %d\n", thread_data->num_elements); 
    printf("Address of vector A on heap: %p\n", &(thread_data->vector_a));
    printf("Address of vector B on heap: %p\n", &(thread_data->vector_b));
    printf("Offset within the vectors for thread: %d\n", thread_data->offset);
    printf("Chunk size to operate on: %d\n", thread_data->chunk_size);
    printf("\n");
}

