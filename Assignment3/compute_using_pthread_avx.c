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
typedef struct {
    int start_row; // where to start
    int end_row; // where to end
    int num_cols; // number of columns
    float *A_elements; // vector A
    float *src; // src vector
    float *dest; // unknowns vector
    float *B_elements; // vecot B
} ThreadData;

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads + AVX.
 * Result must be placed in pthread_avx_solution_x. */
void compute_using_pthread_avx(const matrix_t A, matrix_t pthread_avx_solution_x, const matrix_t B, int max_iter, int num_threads)
{

    int i;
    int num_rows = A.num_rows;
    int num_cols = A.num_columns;
    /* Allocate n x 1 matrix to hold iteration values. */
    matrix_t new_x = allocate_matrix(num_rows, 1, 0);

    /* Initialize current Jacobi solution. */
    for (i = 0; i < num_rows; i++)
    pthread_avx_solution_x.elements[i] = B.elements[i];

    /* Setup the ping-pong buffers */
    float *src = pthread_avx_solution_x.elements;
    float *dest = new_x.elements;
    float *temp;

    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    int done = 0;
    double ssd, mse;
    int num_iter = 0;

    while (!done) {
        // Split the work across multiple threads
        int chunk_size = num_rows / num_threads;
        for (i = 0; i < num_threads; i++) {
            thread_data[i].start_row = i * chunk_size;
            thread_data[i].end_row = (i == num_threads - 1) ? num_rows : (i + 1) * chunk_size;
            thread_data[i].num_cols = num_cols;
            thread_data[i].A_elements = A.elements;
            thread_data[i].src = src;
            thread_data[i].dest = dest;
            thread_data[i].B_elements = B.elements;

            pthread_create(&threads[i], NULL, iterative_jacobian, &thread_data[i]);
        }

        // Join all threads
        for (i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }

        /* Check for convergence and update the unknowns. */
        ssd = 0.0;
        for (i = 0; i < num_rows; i++) {
            float diff = dest[i] - src[i];
            ssd += diff * diff;
        }

        num_iter++;
        mse = sqrt(ssd);  // Mean squared error
        //fprintf(stderr, "Iteration: %d. MSE = %f\n", num_iter, mse); 

        if ((mse <= THRESHOLD) || (num_iter == max_iter))
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

}

void *iterative_jacobian(void *args)
{
    ThreadData *data = (ThreadData *)args;
    int i, j;
    int num_cols = data->num_cols;
    __m256 sum, a, x, tmp, loaded_a_vec;
    float *A, *src, *dest, *B;
    A = data->A_elements;    // Vector A
    src = data->src; // src vector
    dest = data->dest; // unknowns vector
    B = data->B_elements; // Vector B

    for (i = data->start_row; i < data->end_row; i++) {
        sum = _mm256_setzero_ps();  // Initialize sum vector to zero
        a = _mm256_set1_ps(A[i * num_cols + i]);  // Broadcast A[i]

        for (j = 0; j < num_cols; j += VECTOR_SIZE) {
            loaded_a_vec = _mm256_loadu_ps(&A[i * num_cols + j]);  // Load A[i] with unaligned memory accounted for
            x = _mm256_loadu_ps(&src[j]);  // Load the src vector
            tmp = _mm256_mul_ps(loaded_a_vec, x);  // A[i] * src
            sum = _mm256_add_ps(sum, tmp);  // Accumulate sum
        }

        // Horizontal sum of the elements in the sum vector using adjacent addition. Then cast to 32-bit floating point value
        __m128 partial_sum = _mm_add_ps(_mm256_extractf128_ps(sum, 1), _mm256_castps256_ps128(sum));
        partial_sum = _mm_hadd_ps(partial_sum, partial_sum);
        partial_sum = _mm_hadd_ps(partial_sum, partial_sum);
        float new_sum = _mm_cvtss_f32(partial_sum);

        // Subtract the diagonal contribution
        new_sum -= A[i * num_cols + i] * src[i];
        /* Update values for the unknowns for the current row. */
        dest[i] = (B[i] - new_sum) / A[i * num_cols + i];
    }

    pthread_exit(NULL);

}

