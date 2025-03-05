/* GPU kernel to estimate integral of the provided function using the trapezoidal rule. */

/* Device function which implements the function. Device functions can be called from within other __device__ functions or __global__ functions (the kernel), but cannot be called from the host. */ 

__device__ float fd(float x) 
{
     return sqrtf((1 + x*x)/(1 + x*x*x*x));
}

/* Kernel function */
__global__ void trap_kernel(float a, float b, int n, float h) 
{
     extern __shared__ float shared_sums[]; // shared memory for partial sums
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    shared_sums[tid] = 0.0f;

    // Stride over trapezoids and compute partial sums
    for (int i = idx; i < n; i += stride) {
        float x = a + i * h;
        shared_sums[tid] += (fd(x) + fd(x + h)) * h / 2.0f;
    }
    __syncthreads();

    // Tree-style reduction within the thread block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] += shared_sums[tid + s];
        }
        __syncthreads();
    }

    // Accumulate the reduced value into global memory
    if (tid == 0) {
        atomicAdd(global_sum, shared_sums[0]);
    }

}
