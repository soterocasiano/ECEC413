/* Optimize Jacobi using pthread and AVX instructions */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <immintrin.h>
#include <unistd.h>
#include <sched.h>
#include "jacobi_solver.h"

#define VECTOR_SIZE 8 /* AVX operates on 8 single-precision floating-point values */

/* Data structure defining what to pass to each worker thread */
typedef struct thread_data_s {
    int tid;                        /* The thread ID: 0, 1, ... */
    int num_threads;                /* Number of threads in the pool */
    int max_iter;           
    matrix_t A;                     /* Matrix A */
    matrix_t X;                /* Matrix X */
    matrix_t B;
    int offset;                     /* Starting offset for each thread within the vectors. Offsets are specified in terms of mini chunks */ 
    int num_mini_chunks;            /* Number of mini chunks assigned to each thread. Each mini chunk consists of 8 floating point elements */
    double ssd;                     /* Address of the shared variable for the sum */
    pthread_mutex_t *mutex_for_ssd; /* Location of lock variable protecting sum */
} thread_data_t;

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads + AVX.
 * Result must be placed in pthread_avx_solution_x. */
void compute_using_pthread_avx(const matrix_t A, matrix_t pthread_avx_solution_x, const matrix_t B, int max_iter, int num_threads)
{

    pthread_t *thread_id = (pthread_t *)malloc(num_threads * sizeof(pthread_t));                /* Data structure to store thread IDs */
    pthread_attr_t attributes;                                                                  /* Thread attributes */
    pthread_attr_init(&attributes);                                                             /* Initialize thread attributes with default values */

    float ssd = 0.0;                                                                            /* Shared variable for ssd */
    pthread_mutex_t mutex_for_ssd;                                                              /* Lock for the shared variable ssd */
    pthread_mutex_init(&mutex_for_ssd, NULL);                                                   /* Initialize the mutex */

    /* Fork point: Allocate heap memory for required data structures and create the worker threads */
    int i;
    thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);
    int num_mini_chunks_per_thread = (A.num_rows/VECTOR_SIZE)/num_threads;                    /* Calculate number of mini chunks, in size of 8 elements, to assign to each thread */
    //printf("Assigning %d mini chunks to each thread\n", num_mini_chunks_per_thread);
    
    for (i = 0; i < num_threads; i++) {
        thread_data[i].tid = i; 
        thread_data[i].num_threads = num_threads;
        thread_data[i].max_iter = max_iter; 
        thread_data[i].A = A; 
        thread_data[i].B = B;
        thread_data[i].X = pthread_avx_solution_x;
        thread_data[i].offset = i * num_mini_chunks_per_thread;                             /* Calculate the offset for each thread, in terms of mini chnks (eight elements each) */ 
        thread_data[i].num_mini_chunks = num_mini_chunks_per_thread;
        thread_data[i].ssd = ssd;
        thread_data[i].mutex_for_ssd = &mutex_for_ssd;
    }

    for (i = 0; i < num_threads; i++)
        pthread_create(&thread_id[i], &attributes, iterative_jacobian, (void *)&thread_data[i]);
					 
    /* Join point: wait for the workers to finish */
    for (i = 0; i < num_threads; i++)
        pthread_join(thread_id[i], NULL);
		 
    /* Free dynamically allocated data structures */
    free((void *)thread_data);
}

void *iterative_jacobian(void *args)
{
    thread_data_t *thread_data = (thread_data_t *)args;                                 /* Typecast argument to pointer to thread_data_t structure */

    int i, j, k;
    int num_rows = thread_data->A.num_rows;
    int num_cols = thread_data->A.num_columns;

    /* Allocate n x 1 matrix to hold iteration values.*/
    matrix_t new_x = allocate_matrix(num_rows, 1, 0);      
    
    /* Initialize current jacobi solution. */
    for (i = 0; i < num_rows; i++)
        thread_data->X.elements[i] = thread_data->B.elements[i];

    /* Setup the ping-pong buffers */
    float *src = thread_data->X.elements;
    float *dest = new_x.elements;
    float *temp;

    /* Perform Jacobi iteration. */
    int done = 0;
    double ssd, mse;
    int num_iter = 0;
    ssd = 0.0; 
    __m256 a, b, x, tmp;

    while (!done) {
        k = thread_data->tid;
        while (k < thread_data->num_mini_chunks){
            for (i = 0; i < thread_data->num_mini_chunks; i++) {
                //double sum = -A.elements[i * num_cols + i] * src[i];
                __m256 sum = _mm256_setzero_ps();
                for (j = 0; j < thread_data->num_mini_chunks; j++) {
                    if (i == j)
                        continue;
                    else{
                        a = _mm256_set1_ps(thread_data->A.elements[i * thread_data->num_mini_chunks + j]);
                        x = _mm256_load_ps(&src[VECTOR_SIZE * j]);
                        tmp = _mm256_mul_ps(a, x);
                        sum = _mm256_add_ps(sum, tmp);
                        //sum += A.elements[i * num_cols + j] * src[j];
                    }
                }
               
                /* Update values for the unkowns for the current row. */
                b = _mm256_set1_ps(thread_data->B.elements[i]);
                a = _mm256_set1_ps(thread_data->A.elements[i * num_cols + i]);
                tmp = _mm256_sub_ps(b, sum);
                tmp = _mm256_div_ps(tmp, a);
                _mm256_store_ps(&dest[VECTOR_SIZE * i], tmp);
                //dest[i] = (B.elements[i] - sum)/A.elements[i * num_cols + i];
            }
    
            /* Check for convergence and update the unknowns. */
            for (i = 0; i < num_rows; i++) {
                ssd += (dest[i] - src[i]) * (dest[i] - src[i]);
            }
            k = k + thread_data->num_threads;
        }
        

        num_iter++;
        //fprintf(stderr, "Iteration: %d. MSE = %f\n", num_iter, mse); 
        pthread_mutex_lock(thread_data->mutex_for_ssd);
        thread_data->ssd += ssd;
        pthread_mutex_unlock(thread_data->mutex_for_ssd);
        mse = sqrt(thread_data->ssd); /* Mean squared error. */
        if ((mse <= THRESHOLD) || (num_iter == thread_data->max_iter))
            done = 1;
        
        /* Flip the ping-pong buffers */
        temp = src;
        src = dest;
        dest = temp;
    }

    /*if (num_iter < max_iter)
        fprintf(stderr, "\nConvergence achieved after %d iterations\n", num_iter);
    else
        fprintf(stderr, "\nMaximum allowed iterations reached\n");*/

    free(new_x.elements);
		  
    pthread_exit(NULL);
}

