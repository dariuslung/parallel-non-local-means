#ifndef __GPU_GLOBAL_MEM_CUH__
#define __GPU_GLOBAL_MEM_CUH__

#include "utils.cuh"

namespace gpu_global_mem
{
    // Global Kernel: Located outside class to be callable by CUDA runtime
    __global__ void k_nlm_global(float* d_img, float* d_weights, int width, int p_size, float sigma, float* d_out)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        
        if (tid >= width * width)
        {
            return;
        }

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
                float dist = util::calc_patch_dist(d_img, d_weights, width, p_size,
                                                   r_start, c_start,
                                                   i - p_size / 2, j - p_size / 2);
                                                   
                float weight = util::calc_exponent_weight(dist, sigma);
                
                w_sum += weight;
                res_val += weight * d_img[i * width + j];
            }
        }

        d_out[tid] = res_val / w_sum;
    }

    class GpuGlobalProcessor
    {
    private:
        util::NlmParams params;
        float *dev_img_buf = nullptr;
        float *dev_weight_buf = nullptr;
        float *dev_out_buf = nullptr;

        void allocate_resources(const std::vector<float>& host_weights, const float* host_img)
        {
            size_t img_bytes = params.img_width * params.img_width * sizeof(float);
            size_t w_bytes = params.patch_size * params.patch_size * sizeof(float);

            gpu_err_chk(cudaMalloc((void**)&dev_img_buf, img_bytes));
            gpu_err_chk(cudaMalloc((void**)&dev_weight_buf, w_bytes));
            gpu_err_chk(cudaMalloc((void**)&dev_out_buf, img_bytes));

            gpu_err_chk(cudaMemcpy(dev_img_buf, host_img, img_bytes, cudaMemcpyHostToDevice));
            gpu_err_chk(cudaMemcpy(dev_weight_buf, host_weights.data(), w_bytes, cudaMemcpyHostToDevice));
        }

        void free_resources()
        {
            if (dev_img_buf) cudaFree(dev_img_buf);
            if (dev_weight_buf) cudaFree(dev_weight_buf);
            if (dev_out_buf) cudaFree(dev_out_buf);
        }

    public:
        GpuGlobalProcessor(int n, int ps, float psig, float fsig)
        {
            params = {n, ps, psig, fsig};
        }

        ~GpuGlobalProcessor()
        {
            // RAII destructor not fully utilized here due to manual control requirement in this context,
            // but good practice to ensure cleanup.
        }

        std::vector<float> run(float* image_data)
        {
            std::cout << "CUDA Total Threads: " << params.img_width * params.img_width << std::endl;

            std::vector<float> result_host(params.img_width * params.img_width);
            std::vector<float> weights = util::generate_gaussian_kernel(params.patch_size, params.patch_sigma);

            allocate_resources(weights, image_data);
            util::Timer timer(true);
            timer.start("NLM Calculation in GPU Global Memory");
            // Launch configuration:
            // Grid size = n (one block per row)
            // Block size = n (one thread per col)
            // Note: This assumes n <= 1024 (max threads per block).
            k_nlm_global<<<params.img_width, params.img_width>>>(dev_img_buf, 
                                                                 dev_weight_buf, 
                                                                 params.img_width, 
                                                                 params.patch_size, 
                                                                 params.filter_sigma, 
                                                                 dev_out_buf);
            
            gpu_err_chk(cudaPeekAtLastError());
            gpu_err_chk(cudaDeviceSynchronize());
            timer.stop();

            size_t img_bytes = params.img_width * params.img_width * sizeof(float);
            gpu_err_chk(cudaMemcpy(result_host.data(), dev_out_buf, img_bytes, cudaMemcpyDeviceToHost));

            free_resources();
            return result_host;
        }
    };

    // Wrapper
    inline std::vector<float> filter_image(float* image, int n, int patch_size, float patch_sigma, float filter_sigma)
    {
        GpuGlobalProcessor engine(n, patch_size, patch_sigma, filter_sigma);
        return engine.run(image);
    }
}

#endif // __GPU_GLOBAL_MEM_CUH__
