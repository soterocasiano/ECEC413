/* Solve Jacobi using AVX instructions */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "jacobi_solver.h"

#define VECTOR_SIZE 8 

/* FIXME: Complete this function to perform the Jacobi calculation using AVX. 
 * Result must be placed in avx_solution_x. */
void compute_using_avx(const matrix_t A, matrix_t avx_solution_x, const matrix_t B, int max_iter)
{
    int i, j;
    int num_rows = A.num_rows;
    int num_cols = A.num_columns;
    int num_chunks = num_rows/VECTOR_SIZE;

    /* Allocate n x 1 matrix to hold iteration values.*/
    matrix_t new_x = allocate_matrix(num_rows, 1, 0);      
    
    /* Initialize current jacobi solution. */
    for (i = 0; i < num_rows; i++)
        avx_solution_x.elements[i] = B.elements[i];

    /* Setup the ping-pong buffers */
    float *src = avx_solution_x.elements;
    float *dest = new_x.elements;
    float *temp;

    /* Perform Jacobi iteration. */
    int done = 0;
    double ssd, mse;
    int num_iter = 0;

    __m256 a, b, x, tmp;

    while (!done) {
        // AVX in this loop?
        for (i = 0; i < num_chunks; i++) {
            //double sum = -A.elements[i * num_cols + i] * src[i];
            __m256 sum = _mm256_setzero_ps();
            for (j = 0; j < num_chunks; j++) {
                if (i == j)
                    continue;
                else{
                    a = _mm256_set1_ps(A.elements[i * num_chunks + j]);
                    x = _mm256_load_ps(&src[VECTOR_SIZE * j]);
                    tmp = _mm256_mul_ps(a, x);
                    sum = _mm256_add_ps(sum, tmp);
                    //sum += A.elements[i * num_cols + j] * src[j];
                }
            }
           
            /* Update values for the unkowns for the current row. */
            b = _mm256_set1_ps(B.elements[i]);
            a = _mm256_set1_ps(A.elements[i * num_cols + i]);
            tmp = _mm256_sub_ps(b, sum);
            tmp = _mm256_div_ps(tmp, a);
            _mm256_store_ps(&dest[VECTOR_SIZE * i], tmp);
            //dest[i] = (B.elements[i] - sum)/A.elements[i * num_cols + i];
        }

        /* Check for convergence and update the unknowns. */
        ssd = 0.0; 
        for (i = 0; i < num_rows; i++) {
            ssd += (dest[i] - src[i]) * (dest[i] - src[i]);
        }

        num_iter++;
        mse = sqrt(ssd); /* Mean squared error. */
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


