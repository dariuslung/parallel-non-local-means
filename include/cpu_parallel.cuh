#ifndef __CPU_PARALLEL_CUH__
#define __CPU_PARALLEL_CUH__

#include <utils.cuh>
#include <omp.h>
#include <vector>

namespace cpu_parallel
{

// Helper function to filter a single pixel. 
// Identical logic to the serial version, kept here for self-containment.
float filter_pixel(float * image,
                   float * weights,
                   int n,
                   int patch_size,
                   int pixel_row,
                   int pixel_col,
                   float sigma)
{
    float res = 0;
    float sum_w = 0; 
    float dist;
    float w;
    int patch_row_start = pixel_row - patch_size / 2;
    int patch_col_start = pixel_col - patch_size / 2;

    // This inner loop remains serial because it's the granular task for one thread
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

    return res;
}

std::vector<float> filter_image(float * image,
                                int n,
                                int patch_size,
                                float patch_sigma,
                                float filter_sigma)
{
    std::vector<float> res(n * n);
    
    // weights are read-only, so they can be shared across threads safely
    std::vector<float> weights = util::compute_inside_weights(patch_size, patch_sigma);

    // OpenMP Parallelization
    // collapse(2): Flattens the i and j loops into a single loop for better workload distribution
    // schedule(dynamic): Assigns chunks of iterations to threads dynamically, helping balance load
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // Each thread computes a unique index [i * n + j], so writing to 'res' is thread-safe
            res[i * n + j] = filter_pixel(image, weights.data(), n, patch_size, i, j, filter_sigma);
        }
    }

    return res;
}

} // namespace cpu_parallel

#endif // __CPU_PARALLEL_CUH__