#ifndef __CPU_SERIAL_CUH__
#define __CPU_SERIAL_CUH__

#include "utils.cuh"

namespace cpu_serial
{
    class SerialNlm
    {
    private:
        util::NlmParams params;
        std::vector<float> gaussian_weights;

        float compute_single_pixel(const float* img_ptr, int r, int c)
        {
            float result_acc = 0.0f;
            float weight_acc = 0.0f;
            
            int n = params.img_width;
            int ps = params.patch_size;
            int half_ps = ps / 2;

            int p_curr_r = r - half_ps;
            int p_curr_c = c - half_ps;
            
            // Iterate over entire image to find similar patches
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    int p_other_r = i - half_ps;
                    int p_other_c = j - half_ps;

                    float dist = util::calc_patch_dist(img_ptr,
                                                       gaussian_weights.data(),
                                                       n, ps,
                                                       p_curr_r, p_curr_c,
                                                       p_other_r, p_other_c);

                    float w = util::calc_exponent_weight(dist, params.filter_sigma);
                    
                    weight_acc += w;
                    result_acc += w * img_ptr[i * n + j];
                }
            }

            return result_acc / weight_acc;
        }

    public:
        SerialNlm(int n, int patch_size, float p_sigma, float f_sigma)
        {
            params = {n, patch_size, p_sigma, f_sigma};
            gaussian_weights = util::generate_gaussian_kernel(patch_size, p_sigma);
        }

        std::vector<float> process(const std::vector<float>& input_img)
        {
            // Convert to raw pointer for internal processing
            const float* raw_img = input_img.data();
            std::vector<float> output_img(params.img_width * params.img_width);
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < params.img_width; i++)
            {
                for (int j = 0; j < params.img_width; j++)
                {
                    output_img[i * params.img_width + j] = compute_single_pixel(raw_img, i, j);
                }
            } 
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "NLM Calculation for entire image (" << params.img_width << "x" << params.img_width << ") took: " << duration.count() / 1000.0 << " ms" << std::endl; 
            return output_img;
        }
    };

    // Wrapper function to maintain compatibility with original main code
    inline std::vector<float> filter_image(float* image, int n, int patch_size, float patch_sigma, float filter_sigma)
    {
        // Copy raw pointer to vector for the class interface
        std::vector<float> img_vec(image, image + n * n);
        
        SerialNlm processor(n, patch_size, patch_sigma, filter_sigma);
        return processor.process(img_vec);
    }
}

#endif // __CPU_SERIAL_CUH__