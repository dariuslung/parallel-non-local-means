#ifndef __CUDAFILTERINGGLOBALMEM_CUH__
#define __CUDAFILTERINGGLOBALMEM_CUH__

#include <utils.cuh>

namespace gpu_global_mem
{

__global__ void filter_pixel(float * image,
                             float * weights,
                             int n,
                             int patch_size,
                             float sigma,
                             float *filtered_image)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= n * n)
    {
        return;
    }

    int pixel_row = blockIdx.x;
    int pixel_col = threadIdx.x;

    float res = 0;
    float sum_w = 0; // sum_w is the Z(i) of w(i, j) formula
    float dist;
    float w;
    int patch_row_start = pixel_row - patch_size / 2;
    int patch_col_start = pixel_col - patch_size / 2;

    __syncthreads();

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            dist = util::compute_patch_distance(image,
                                                weights,
                                                n,
                                                patch_size,
                                                patch_row_start,
                                                patch_col_start,
                                                i - patch_size / 2,
                                                j - patch_size / 2);
            w = util::compute_weight(dist, sigma);
            sum_w += w;
            res += w * image[i * n + j];
        }
    }
    res = res / sum_w;

    filtered_image[index] = res;
}

std::vector<float> filter_image(float * image,
                                int n,
                                int patch_size,
                                float patch_sigma,
                                float filter_sigma)
{
    std::vector<float> res(n * n);
    
    std::vector<float> weights = util::compute_inside_weights(patch_size, patch_sigma);

    int size_image = n * n * sizeof(float);
    int size_weights = patch_size * patch_size * sizeof(float);

    float *d_image, *d_weights, *d_res;

    gpu_err_chk(cudaMalloc((void **)&d_image, size_image));
    gpu_err_chk(cudaMalloc((void **)&d_weights, size_weights));
    gpu_err_chk(cudaMalloc((void **)&d_res, size_image));

    cudaMemcpy(d_image, image, size_image, cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_weights, weights.data(), size_weights, cudaMemcpyHostToDevice);

    filter_pixel<<<n, n>>>(d_image, d_weights, n, patch_size, filter_sigma, d_res);

    cudaMemcpy(res.data(), d_res, size_image, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_weights);
    cudaFree(d_res);

    return res;
}

} // namespace gpu_global_mem

#endif // __CUDAFILTERINGGLOBALMEM_CUH__