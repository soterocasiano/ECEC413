/* ld and execute as follows:
        make clean && make
        ./trap a b n

 * Author: Naga Kandasamy
 * Date modified: February 28, 2025

 * Student name(s): Sotero Casiano, Jeffrey Lau
 * Date modified: 3/5/2025
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

/* Include the kernel code */
#include "trap_kernel.cu"

double compute_on_device(float, float, int, float);
extern "C" double compute_gold(float, float, int, float);

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s a b n\n", argv[0]);
        fprintf(stderr, "a: Start limit. \n");
        fprintf(stderr, "b: end limit\n");
        fprintf(stderr, "n: number of trapezoids\n");
        exit(EXIT_FAILURE);
    }

    int a = atoi(argv[1]); /* Left limit */
    int b = atoi(argv[2]); /* Right limit */
    int n = atoi(argv[3]); /* Number of trapezoids */

    float h = (b-a)/(float)n; // Height of each trapezoid
        printf("Number of trapezoids = %d\n", n);
    printf("Height of each trapezoid = %f \n", h);
        double cpu_start_time = get_time();
        double reference = compute_gold(a, b, n, h);
        double cpu_end_time = get_time();
        printf("Reference solution computed on the CPU = %f \n", reference);
        printf("CPU time = %f seconds\n", cpu_end_time - cpu_start_time);
        /* Write this function to complete the trapezoidal on the GPU. */
        double gpu_start_time = get_time();
        double gpu_result = compute_on_device(a, b, n, h);
        double gpu_end_time = get_time();
        printf("Solution computed on the GPU = %f \n", gpu_result);
        printf("GPU time = %f seconds\n", gpu_end_time - gpu_start_time);
}

/* Complete this function to perform the trapezoidal rule on the GPU. */
double compute_on_device(float a, float b, int n, float h)
{
    float *d_global_sum;
    float h_global_sum = 0.0f;

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_global_sum, sizeof(float));
    cudaMemcpy(d_global_sum, &h_global_sum, sizeof(float), cudaMemcpyHostToDevice);

    // Define thread block and grid sizes
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    trap_kernel<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(a, b, n, h, d_global_sum);
    cudaDeviceSynchronize();
    // Copy the result back to the host
    cudaMemcpy(&h_global_sum, d_global_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_global_sum);

    return (double)h_global_sum;
}
