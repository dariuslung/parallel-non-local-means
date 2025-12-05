#ifndef __GPU_GLOBAL_MEM_CUH__
#define __GPU_GLOBAL_MEM_CUH__

#include "utils.cuh"

namespace gpu_global_mem
{
    // Device-specific helper: Computes patch distance with optional intrinsics
    template <bool UseIntrinsics>
    __device__ __forceinline__ float calc_patch_dist(const float * img_data,
                                                               const float * weight_kernel,
                                                               int width,
                                                               int p_size,
                                                               int r1, int c1,
                                                               int r2, int c2)
    {
        float diff_sum = 0.0f;
        
        for (int i = 0; i < p_size; i++)
        {
            for (int j = 0; j < p_size; j++)
            {
                bool p1_in = util::check_bound(width, r1 + i, c1 + j);
                bool p2_in = util::check_bound(width, r2 + i, c2 + j);

                if (p1_in && p2_in)
                {
                    int idx1 = (r1 + i) * width + (c1 + j);
                    int idx2 = (r2 + i) * width + (c2 + j);
                    int w_idx = i * p_size + j;
                    
                    float val1 = img_data[idx1];
                    float val2 = img_data[idx2];
                    float diff = val1 - val2;
                    float spat_w = weight_kernel[w_idx];

                    if constexpr (UseIntrinsics) 
                    {
                        // Fused Multiply-Add intrinsic
                        diff_sum = __fmaf_rn(diff * diff, spat_w, diff_sum);
                    } 
                    else 
                    {
                        // Standard math for non-intrinsic GPU mode
                        diff_sum += spat_w * diff * diff;
                    }
                }
            }
        }
        return diff_sum;
    }

    // Global Kernel with Intrinsics Toggle
    template <bool UseIntrinsics>
    __global__ void k_nlm_global(float* d_img, 
                                 float* d_host_weights, 
                                 int width, 
                                 int p_size, 
                                 float neg_inv_sigma_sq, 
                                 float* d_out)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= width * width) return;

        int r_idx = blockIdx.x; // Row derived from Block ID
        int c_idx = threadIdx.x; // Col derived from Thread ID within block

        // Re-calculate row/col based on linear ID to be safe if dimensions change
        // But following original logic: 
        // Note: Original code mapped Block->Row and Thread->Col. 
        // This implies gridDim = n, blockDim = n.
        
        int r_start = r_idx - p_size / 2;
        int c_start = c_idx - p_size / 2;

        float res_val = 0.0f;
        float w_sum = 0.0f;

        // Iterate entire image
        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < width; j++)
            {
                // Use templated distance function
                float dist = calc_patch_dist<UseIntrinsics>(d_img, d_host_weights, width, p_size,
                                                                            r_start, c_start,
                                                                            i - p_size / 2, j - p_size / 2);
                                                   
                float weight = util::compute_weight_metric<UseIntrinsics>(dist, neg_inv_sigma_sq);
                
                w_sum += weight;
                res_val = util::accumulate_pixel<UseIntrinsics>(weight, d_img[i * width + j], res_val);
            }
        }

        d_out[tid] = res_val / w_sum;
    }

    class GpuGlobalProcessor
    {
    private:
        util::NlmParams params;
        float *d_img_ptr = nullptr;
        float *d_host_weights_ptr = nullptr;
        float *d_out_ptr = nullptr;

        void allocate_resources(const std::vector<float>& host_host_weights, const float* host_img)
        {
            size_t bytes_img = params.img_width * params.img_width * sizeof(float);
            size_t bytes_host_weights = params.patch_size * params.patch_size * sizeof(float);

            gpu_err_chk(cudaMalloc((void**)&d_img_ptr, bytes_img));
            gpu_err_chk(cudaMalloc((void**)&d_host_weights_ptr, bytes_host_weights));
            gpu_err_chk(cudaMalloc((void**)&d_out_ptr, bytes_img));

            gpu_err_chk(cudaMemcpy(d_img_ptr, host_img, bytes_img, cudaMemcpyHostToDevice));
            gpu_err_chk(cudaMemcpy(d_host_weights_ptr, host_host_weights.data(), bytes_host_weights, cudaMemcpyHostToDevice));
        }

        void free_resources()
        {
            if (d_img_ptr) cudaFree(d_img_ptr);
            if (d_host_weights_ptr) cudaFree(d_host_weights_ptr);
            if (d_out_ptr) cudaFree(d_out_ptr);
        }

    public:
        GpuGlobalProcessor(int n, int ps, float psig, float fsig)
        {
            params = {n, ps, psig, fsig};
        }

        ~GpuGlobalProcessor() { free_resources(); }

        std::vector<float> run(float* image_data, bool use_intrinsics)
        {
            std::vector<float> host_result(params.img_width * params.img_width);
            std::vector<float> host_weights = util::generate_gaussian_kernel(params.patch_size, params.patch_sigma);

            allocate_resources(host_weights, image_data);
            
            float neg_inv_sigma_sq = -1.0f / (params.filter_sigma * params.filter_sigma);

            util::Timer timer(true);
            timer.start("NLM Calculation in GPU Global Memory");

            if (use_intrinsics)
            {
                k_nlm_global<true><<<params.img_width, params.img_width>>>(d_img_ptr, d_host_weights_ptr, 
                    params.img_width, params.patch_size, neg_inv_sigma_sq, d_out_ptr);
            }
            else
            {
                k_nlm_global<false><<<params.img_width, params.img_width>>>(d_img_ptr, d_host_weights_ptr, 
                    params.img_width, params.patch_size, neg_inv_sigma_sq, d_out_ptr);
            }
            
            gpu_err_chk(cudaPeekAtLastError());
            gpu_err_chk(cudaDeviceSynchronize());
            timer.stop();

            size_t bytes_img = params.img_width * params.img_width * sizeof(float);
            gpu_err_chk(cudaMemcpy(host_result.data(), d_out_ptr, bytes_img, cudaMemcpyDeviceToHost));

            return host_result;
        }
    };

    inline std::vector<float> filter_image(float* image, int n, int patch_size, float patch_sigma, float filter_sigma, bool use_intrinsics)
    {
        GpuGlobalProcessor processor(n, patch_size, patch_sigma, filter_sigma);
        return processor.run(image, use_intrinsics);
    }
}

#endif // __GPU_GLOBAL_MEM_CUH__