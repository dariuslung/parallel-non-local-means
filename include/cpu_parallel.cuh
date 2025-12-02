#ifndef __CPU_PARALLEL_CUH__
#define __CPU_PARALLEL_CUH__

#include "utils.cuh"
#include <omp.h>

namespace cpu_parallel
{
    class ParallelNlm
    {
    private:
        util::NlmParams params;
        std::vector<float> gaussian_weights;

        // Internal helper for pixel calculation
        // Marked const to ensure thread safety (read-only on members)
        float solve_pixel(const float* img_ptr, int r, int c) const
        {
            float result_accum = 0.0f;
            float total_weight = 0.0f;
            
            int n = params.img_width;
            int ps = params.patch_size;
            int offset = ps / 2;

            int r_start = r - offset;
            int c_start = c - offset;

            // Inner loop remains serial as it represents the atomic task for one thread
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float dist = util::calc_patch_dist(img_ptr,
                                                       gaussian_weights.data(),
                                                       n, ps,
                                                       r_start, c_start,
                                                       i - offset, j - offset);

                    float w = util::calc_exponent_weight(dist, params.filter_sigma);
                    
                    total_weight += w;
                    result_accum += w * img_ptr[i * n + j];
                }
            }
            return result_accum / total_weight;
        }

    public:
        ParallelNlm(int n, int patch_size, float p_sigma, float f_sigma)
        {
            params = {n, patch_size, p_sigma, f_sigma};
            gaussian_weights = util::generate_gaussian_kernel(patch_size, p_sigma);
        }

        std::vector<float> execute(float* input_data)
        {
            int n = params.img_width;
            std::vector<float> result(n * n);

            // OMP Directive:
            // collapse(2) merges loops for better granularity
            // schedule(dynamic) helps if some pixels take longer (e.g. boundary checks)
            auto start_time = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    // Thread-safe write to distinct memory locations
                    result[i * n + j] = solve_pixel(input_data, i, j);
                }
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "NLM Calculation for entire image (" << params.img_width << "x" << params.img_width << ") took: " << duration.count() / 1000.0 << " ms" << std::endl; 
            return result;
        }
    };

    // Standardized interface wrapper
    inline std::vector<float> filter_image(float* image, int n, int patch_size, float patch_sigma, float filter_sigma)
    {
        ParallelNlm processor(n, patch_size, patch_sigma, filter_sigma);
        return processor.execute(image);
    }
}

#endif // __CPU_PARALLEL_CUH__