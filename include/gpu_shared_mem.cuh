#ifndef __GPU_SHARED_MEM_CUH__
#define __GPU_SHARED_MEM_CUH__

#include "utils.cuh"

// Dynamic shared memory declaration
extern __shared__ float shared_patch_cache[];

namespace gpu_shared_mem
{
    // Device-specific helper: Accesses Shared Memory "shared_patch_cache"
    // This replaces the function previously in utils.cuh to decouple dependencies.
    __device__ float calc_dist_shared(const float* global_img,
                                      const float* weights,
                                      int width,
                                      int p_size,
                                      int r1, int c1, // Coords for patch 1 (in Shared Mem)
                                      int r2, int c2) // Coords for patch 2 (in Global Mem)
    {
        float* s_cache = shared_patch_cache;
        float diff_accum = 0.0f;

        for (int i = 0; i < p_size; i++)
        {
            for (int j = 0; j < p_size; j++)
            {
                // Verify bounds
                bool p1_ok = util::check_bound(width, r1 + i, c1 + j);
                bool p2_ok = util::check_bound(width, r2 + i, c2 + j);

                if (p1_ok && p2_ok)
                {
                    // Access Logic:
                    // Patch 1 is loaded into shared memory. 
                    // Its layout in shared memory corresponds to the current thread/block's row stripe.
                    // The shared memory index logic: i * width + c1 + j is technically mimicking global
                    // but accessed via the 's_cache' pointer.
                    
                    float val1 = s_cache[i * width + c1 + j];
                    float val2 = global_img[(r2 + i) * width + (c2 + j)];
                    
                    float diff = val1 - val2;
                    diff_accum += weights[i * p_size + j] * diff * diff;
                }
            }
        }
        return diff_accum;
    }

    __global__ void k_nlm_shared(float* d_img, float* d_weights, int width, int p_size, float sigma, float* d_out)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= width * width) return;

        int row = blockIdx.x;
        int col = threadIdx.x;

        // --- Shared Memory Loading Phase ---
        float* s_cache = shared_patch_cache;
        
        // Each thread loads needed rows for the patch centered at this row
        // Optimization: Cooperative loading could be better, but sticking to logic:
        for (int i = 0; i < p_size; i++)
        {
            int target_row = row - p_size / 2 + i;
            if (target_row >= 0 && target_row < width)
            {
                // Loading global image data into shared memory buffer
                // Map: [col + i * width] -> simulates a 2D patch strip in 1D shared array
                s_cache[col + i * width] = d_img[col + target_row * width];
            }
        }
        __syncthreads(); // Ensure cache is populated

        // --- Computation Phase ---
        float res_acc = 0.0f;
        float weight_acc = 0.0f;
        
        int p_r_start = row - p_size / 2;
        int p_c_start = col - p_size / 2;

        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < width; j++)
            {
                // Use the local shared-memory helper
                float dist = calc_dist_shared(d_img, d_weights, width, p_size,
                                              p_r_start, p_c_start,
                                              i - p_size / 2, j - p_size / 2);
                                              
                float w = util::calc_exponent_weight(dist, sigma);
                
                weight_acc += w;
                res_acc += w * d_img[i * width + j];
            }
        }
        
        d_out[tid] = res_acc / weight_acc;
    }

    class GpuSharedProcessor
    {
    private:
        util::NlmParams params;
        float *d_img_ptr = nullptr;
        float *d_weights_ptr = nullptr;
        float *d_res_ptr = nullptr;

    public:
        GpuSharedProcessor(int n, int ps, float psig, float fsig)
        {
            params = {n, ps, psig, fsig};
        }

        std::vector<float> compute(float* host_img)
        {
            std::cout << "CUDA Total Threads: " << params.img_width * params.img_width << std::endl;

            size_t bytes_img = params.img_width * params.img_width * sizeof(float);
            size_t bytes_weights = params.patch_size * params.patch_size * sizeof(float);
            
            // Shared memory size calculation:
            // width * patch_size floats are needed per block (per row logic)
            size_t bytes_shared = params.img_width * params.patch_size * sizeof(float);
            
            std::vector<float> host_res(params.img_width * params.img_width);
            std::vector<float> host_weights = util::generate_gaussian_kernel(params.patch_size, params.patch_sigma);

            gpu_err_chk(cudaMalloc(&d_img_ptr, bytes_img));
            gpu_err_chk(cudaMalloc(&d_weights_ptr, bytes_weights));
            gpu_err_chk(cudaMalloc(&d_res_ptr, bytes_img));

            gpu_err_chk(cudaMemcpy(d_img_ptr, host_img, bytes_img, cudaMemcpyHostToDevice));
            gpu_err_chk(cudaMemcpy(d_weights_ptr, host_weights.data(), bytes_weights, cudaMemcpyHostToDevice));
            util::Timer timer(true);
            timer.start("NLM Calculation in GPU Shared Memory");
            k_nlm_shared<<<params.img_width, params.img_width, bytes_shared>>>(d_img_ptr, 
                                                                               d_weights_ptr, 
                                                                               params.img_width, 
                                                                               params.patch_size, 
                                                                               params.filter_sigma, 
                                                                               d_res_ptr);
            
            gpu_err_chk(cudaPeekAtLastError());
            gpu_err_chk(cudaDeviceSynchronize());

            gpu_err_chk(cudaMemcpy(host_res.data(), d_res_ptr, bytes_img, cudaMemcpyDeviceToHost));
            timer.stop();
            
            cudaFree(d_img_ptr);
            cudaFree(d_weights_ptr);
            cudaFree(d_res_ptr);

            return host_res;
        }
    };

    // Wrapper
    inline std::vector<float> filter_image(float* image, int n, int patch_size, float patch_sigma, float filter_sigma)
    {
        GpuSharedProcessor proc(n, patch_size, patch_sigma, filter_sigma);
        return proc.compute(image);
    }
}

#endif // __GPU_SHARED_MEM_CUH__