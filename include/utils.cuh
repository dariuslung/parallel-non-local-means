#ifndef __UTILS_CUH__
#define __UTILS_CUH__

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <iterator>
#include <fstream>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <iomanip>

// Robust CUDA Error checking macro
#define gpu_err_chk(ans) { gpu_assert((ans), __FILE__, __LINE__); }

inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

namespace util
{
    // Constants
    const float EPSILON = 1e-4f;

    // Configuration structure to pass parameters cleanly
    struct NlmParams
    {
        int img_width;
        int patch_size;
        float patch_sigma;
        float filter_sigma;
    };

    // --- Timer Class ---
    class Timer
    {
    private:
        float duration_val;
        bool should_print;
        std::string operation_name;
        std::chrono::high_resolution_clock::time_point time_start, time_end;

    public:
        Timer(bool print_enabled) : should_print(print_enabled), duration_val(0.0f) {}

        void start(const std::string& desc)
        {
            operation_name = desc;
            time_start = std::chrono::high_resolution_clock::now();
        }

        void stop()
        {
            time_end = std::chrono::high_resolution_clock::now();
            duration_val = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count();
            
            if (should_print)
            {
                std::cout << "\n[Timer] " << operation_name << ": " << duration_val / 1e3 << " ms\n" << std::endl;
            }
        }
    };

    // --- Math & Bounds Helpers ---

    __host__ __device__ inline bool check_bound(int limit, int r, int c)
    {
        return (r >= 0 && r < limit && c >= 0 && c < limit);
    }

    __host__ __device__ inline float calc_exponent_weight(float dist_sq, float sigma)
    {
        return expf(-dist_sq / (sigma * sigma));
    }

    // Standard Euclidean distance for patches (Host & Global Memory GPU)
    __host__ __device__ inline float calc_patch_dist(const float * img_data,
                                                     const float * weight_kernel,
                                                     int width,
                                                     int p_size,
                                                     int r1,
                                                     int c1,
                                                     int r2,
                                                     int c2)
    {
        float diff_sum = 0.0f;
        float pixel_diff;
        
        // Iterating flatly or 2D, here we do 2D for clarity in patch logic
        for (int i = 0; i < p_size; i++)
        {
            for (int j = 0; j < p_size; j++)
            {
                // Check bounds for both patches
                bool p1_in = check_bound(width, r1 + i, c1 + j);
                bool p2_in = check_bound(width, r2 + i, c2 + j);

                if (p1_in && p2_in)
                {
                    int idx1 = (r1 + i) * width + (c1 + j);
                    int idx2 = (r2 + i) * width + (c2 + j);
                    int w_idx = i * p_size + j;

                    pixel_diff = img_data[idx1] - img_data[idx2];
                    diff_sum += weight_kernel[w_idx] * pixel_diff * pixel_diff;
                }
            }
        }
        return diff_sum;
    }

    // --- Pre-computation Helpers ---

    inline std::vector<float> generate_gaussian_kernel(int p_size, float p_sigma)
    {
        std::vector<float> kernel(p_size * p_size);
        int center = p_size / 2;
        float total_weight = 0.0f;

        for (int i = 0; i < p_size; i++)
        {
            for (int j = 0; j < p_size; j++)
            {
                float dy = static_cast<float>(center - i);
                float dx = static_cast<float>(center - j);
                float dist_sq = dy*dy + dx*dx;
                
                float val = exp(-dist_sq / (2.0f * p_sigma * p_sigma));
                kernel[i * p_size + j] = val;
                total_weight += val;
            }
        }

        // Normalize
        for (auto& k : kernel)
        {
            k /= total_weight;
        }

        return kernel;
    }

    inline std::vector<float> calc_diff_image(const std::vector<float>& src, const std::vector<float>& filtered, int n)
    {
        if (src.size() != filtered.size()) throw std::runtime_error("Dimension mismatch in residual calc.");

        std::vector<float> residual(n * n);
        for (size_t i = 0; i < src.size(); i++)
        {
            residual[i] = src[i] - filtered[i];
        }
        std::cout << "Residual calculated successfully.\n\n";
        return residual;
    }
}

// IO and Test namespaces simplified for structure
namespace file
{
    inline std::vector<float> read(const std::string& path, int rows, int cols, char delim)
    {
        std::vector<float> buffer(rows * cols);
        std::ifstream file_stream(path);

        if (!file_stream.is_open()) throw std::runtime_error("Cannot open: " + path);

        std::string line_buf;
        for (int i = 0; i < rows; i++)
        {
            if (!std::getline(file_stream, line_buf)) break;
            std::istringstream stream(line_buf);
            std::string token;
            int j = 0;
            while (std::getline(stream, token, delim) && j < cols)
            {
                try { buffer[i * cols + j++] = std::stof(token); }
                catch (...) { buffer[i * cols + j - 1] = 0.0f; }
            }
        }
        return buffer;
    }

    inline std::string write(const std::vector<float>& data, const std::string& name, int r, int c)
    {
        std::string path = "./data/out/" + name + ".txt";
        std::ofstream out_stream(path);
        if (!out_stream.is_open())
        {
            std::cerr << "Failed to write to " << path << std::endl;
            return "";
        }

        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                out_stream << data[i * c + j] << " ";
            }
            out_stream << "\n";
        }
        return path;
    }

    inline std::string write_images(const std::vector<float>& img, const std::vector<float>& res, 
                                    int p_size, float f_sig, float p_sig, int r, int c, int mode_idx)
    {
        std::string suffix = std::to_string(p_size) + "_" + std::to_string(f_sig) + "_" + std::to_string(p_sig);
        std::string mode_str = "unknown";
        
        switch(mode_idx) {
            case 0: mode_str = "cpu_serial"; break;
            case 1: mode_str = "cpu_parallel"; break;
            case 2: mode_str = "gpu_global"; break;
            case 3: mode_str = "gpu_shared"; break;
        }

        std::string f_name = mode_str + "_filtered_image_" + suffix;
        std::string r_name = mode_str + "_residual_" + suffix;

        std::string ret = write(img, f_name, r, c);
        write(res, r_name, r, c);
        
        std::cout << "Files written: " << f_name << " & " << r_name << "\n\n";
        return ret;
    }
}

namespace test
{
    inline void validate(const std::string& gold_path, const std::string& test_path, int n)
    {
        auto gold = file::read(gold_path, n, n, ',');
        auto result = file::read(test_path, n, n, ' ');
        int errs = 0;
        bool success = true;

        for (size_t i = 0; i < gold.size(); i++)
        {
            if (std::abs(gold[i] - result[i]) > util::EPSILON)
            {
                if (errs++ < 10)
                    std::cout << "Mismatch at " << i << ": Ref=" << gold[i] << " Act=" << result[i] << "\n";
                success = false;
            }
        }

        if (success) std::cout << "Validation: PASSED\n\n";
        else std::cout << "Validation: FAILED (" << errs << " errors)\n\n";
    }

    inline float calc_mse(const std::string& gold_path, const std::string& test_path, int n)
    {
        auto gold = file::read(gold_path, n, n, ',');
        auto result = file::read(test_path, n, n, ' ');
        double sum_sq = 0;
        for (size_t i = 0; i < gold.size(); i++)
        {
            sum_sq += std::pow(gold[i] - result[i], 2);
        }
        return static_cast<float>(sum_sq / (n * n));
    }
}

#endif // __UTILS_CUH__