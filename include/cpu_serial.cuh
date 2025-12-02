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
        SerialNlm(int n, int patch_size, float p_sigma, float f_sigma)
        {
            params = {n, patch_size, p_sigma, f_sigma};
            gaussian_weights = util::generate_gaussian_kernel(patch_size, p_sigma);
        }

        std::vector<float> execute(const std::vector<float>& input_img)
        {
            // Convert to raw pointer for internal processing
            const float* raw_img = input_img.data();
            std::vector<float> output_img(params.img_width * params.img_width);
            util::Timer timer(true);
            timer.start("NLM Calculation in CPU Serial");
            for (int i = 0; i < params.img_width; i++)
            {
                for (int j = 0; j < params.img_width; j++)
                {
                    output_img[i * params.img_width + j] = compute_single_pixel(raw_img, i, j);
                }
            } 
            timer.stop();
            return output_img;
        }
    };

    // Wrapper function to maintain compatibility with original main code
    inline std::vector<float> filter_image(float* image, int n, int patch_size, float patch_sigma, float filter_sigma)
    {
        // Copy raw pointer to vector for the class interface
        std::vector<float> img_vec(image, image + n * n);
        
        SerialNlm processor(n, patch_size, patch_sigma, filter_sigma);
        return processor.execute(img_vec);
    }
}

#endif // __CPU_SERIAL_CUH__