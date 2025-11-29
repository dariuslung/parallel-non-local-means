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
#include <stdexcept> // For std::runtime_error
#include <iomanip>   // For formatted output

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

extern __shared__ float s[];

namespace util
{

// Epsilon for floating point comparisons
const float EPSILON = 1e-4f;

/* -------------------------------------------------------------------------- */
/* timer                                   */
/* -------------------------------------------------------------------------- */

class Timer
{
public:
    Timer(bool print) : print(print), duration(0.0f) {}

    void start(const std::string& operation_desc)
    {
        _operation_desc = operation_desc;
        t1 = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        t2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        if (print)
        {
            std::cout << "\n[Timer] " << _operation_desc << ": " << duration / 1e3 << " ms\n" << std::endl;
        }
    }

private:
    float duration;
    bool print;
    std::string _operation_desc;
    std::chrono::high_resolution_clock::time_point t1, t2;
};

/* -------------------------------------------------------------------------- */
/* host & device utils                            */
/* -------------------------------------------------------------------------- */

__host__ __device__ inline bool is_in_bounds(int n, int x, int y)
{
    return x >= 0 && x < n && y >= 0 && y < n;
}

__host__ __device__ inline float compute_weight(float dist, float sigma)
{
    return expf(-dist / (sigma * sigma));
}

// Patch-to-patch euclidean distance
__host__ __device__ float compute_patch_distance(const float * image,
                                                 const float * weights,
                                                 int n,
                                                 int patch_size,
                                                 int p1_row_start,
                                                 int p1_col_start,
                                                 int p2_row_start,
                                                 int p2_col_start)
{
    float ans = 0;
    float temp;

    for (int i = 0; i < patch_size; i++)
    {
        for (int j = 0; j < patch_size; j++)
        {
            // Note: Bounds check logic implies zero-padding behavior if out of bounds (implicit in original logic)
            // Ideally, we should handle boundary conditions explicitly, but keeping original logic:
            if (is_in_bounds(n, p1_row_start + i, p1_col_start + j) &&
                is_in_bounds(n, p2_row_start + i, p2_col_start + j))
            {
                temp = image[(p1_row_start + i) * n + p1_col_start + j] -
                       image[(p2_row_start + i) * n + p2_col_start + j];
                ans += weights[i * patch_size + j] * temp * temp;
            }
        }
    }

    return ans;
}

/* -------------------------------------------------------------------------- */
/* host utils                                 */
/* -------------------------------------------------------------------------- */

// Refactored to return std::vector to manage memory automatically (prevents leaks)
std::vector<float> compute_inside_weights(int patch_size, float patch_sigma)
{
    std::vector<float> weights(patch_size * patch_size);
    int central_pixel_row = patch_size / 2;
    int central_pixel_col = central_pixel_row;
    float dist;
    float sum_w = 0;

    for (int i = 0; i < patch_size; i++)
    {
        for (int j = 0; j < patch_size; j++)
        {
            dist = (float)((central_pixel_row - i) * (central_pixel_row - i) +
                           (central_pixel_col - j) * (central_pixel_col - j));
            weights[i * patch_size + j] = exp(-dist / (2 * (patch_sigma * patch_sigma)));
            sum_w += weights[i * patch_size + j];
        }
    }

    // Normalize
    for (int i = 0; i < patch_size * patch_size; i++)
    {
        weights[i] /= sum_w;
    }

    return weights;
}

std::vector<float> compute_residual(const std::vector<float>& image, const std::vector<float>& filtered_image, int n)
{
    if (image.size() != filtered_image.size())
    {
        throw std::runtime_error("Size mismatch in compute_residual");
    }

    std::vector<float> res(n * n);
    for (int i = 0; i < n * n; i++)
    {
        res[i] = image[i] - filtered_image[i];
    }
    std::cout << "Residual calculated successfully." << std::endl << std::endl;

    return res;
}

/* -------------------------------------------------------------------------- */
/* device utils                                */
/* -------------------------------------------------------------------------- */

__device__ float cuda_compute_patch_distance(const float * image,
                                             const float * weights,
                                             int n,
                                             int patch_size,
                                             int p1_row_start,
                                             int p1_col_start,
                                             int p2_row_start,
                                             int p2_col_start)
{
    // 's' is declared extern __shared__ float s[] at the top
    float *patches = s;

    float ans = 0;
    float temp;

    for (int i = 0; i < patch_size; i++)
    {
        for (int j = 0; j < patch_size; j++)
        {
            if (is_in_bounds(n, p1_row_start + i, p1_col_start + j) &&
                is_in_bounds(n, p2_row_start + i, p2_col_start + j))
            {
                // Accessing shared memory for the first patch, global memory for the second
                temp = patches[i * n + p1_col_start + j] -
                       image[(p2_row_start + i) * n + p2_col_start + j];
                ans += weights[i * patch_size + j] * temp * temp;
            }
        }
    }

    return ans;
}

} // namespace util

namespace prt
{

void row_major_array(const float * arr, int n, int m)
{
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            std::cout << arr[i * m + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::defaultfloat << std::endl;
}

void row_major_vector(const std::vector<float>& vector, int n, int m)
{
    // Reuse array implementation
    row_major_array(vector.data(), n, m);
}

void parameters(int patch_size, float filter_sigma, float patch_sigma)
{
    std::cout << "--------------------------------------" << std::endl
              << "Processing Parameters:"                 << std::endl
              << "  - Patch size:   " << patch_size       << std::endl
              << "  - Patch sigma:  " << patch_sigma      << std::endl
              << "  - Filter Sigma: " << filter_sigma     << std::endl
              << "--------------------------------------" << std::endl << std::endl;
}

} // namespace prt

namespace file
{

std::vector<float> read(const std::string& file_path, int n, int m, char delim)
{
    std::vector<float> image(n * m);
    std::ifstream input(file_path);

    if (!input.is_open())
    {
        throw std::runtime_error("Error: Could not open file " + file_path);
    }

    std::string line;
    for (int i = 0; i < n; i++)
    {
        if (!std::getline(input, line))
        {
            break; // Handle fewer lines than expected
        }

        std::istringstream iss(line);
        std::string num;
        int j = 0;
        while (std::getline(iss, num, delim) && j < m)
        {
            try
            {
                image[i * m + j++] = std::stof(num);
            }
            catch (...)
            {
                // Handle parsing errors or empty tokens
                image[i * m + j - 1] = 0.0f;
            }
        }
    }
    return image;
}

std::string write(const std::vector<float>& image, const std::string& file_name, int row_num, int col_num)
{
    std::string full_path = "./data/out/" + file_name + ".txt";
    std::ofstream output_file(full_path);

    if (!output_file.is_open())
    {
        std::cerr << "Warning: Could not create output file at " << full_path
                  << ". Please ensure './data/out/' directory exists." << std::endl;
        return "";
    }

    for (int i = 0; i < row_num; i++)
    {
        for (int j = 0; j < col_num; j++)
        {
            output_file << image[i * col_num + j] << " ";
        }
        output_file << "\n";
    }

    return full_path;
}

std::string write_images(const std::vector<float>& filtered_image,
                         const std::vector<float>& residual,
                         int patch_size,
                         float filter_sigma,
                         float patch_sigma,
                         int row_num,
                         int col_num,
                         bool use_gpu)
{
    std::string params = std::to_string(patch_size) + "_" +
                         std::to_string(filter_sigma) + "_" +
                         std::to_string(patch_sigma);

    std::string filtered_name = "filtered_image_" + params;
    if (use_gpu)
    {
        filtered_name = "cuda_" + filtered_name;
    }
    std::string ret = file::write(filtered_image, filtered_name, row_num, col_num);

    std::string res_name = "residual_" + params;
    if (use_gpu)
    {
        res_name = "cuda_" + res_name;
    }
    file::write(residual, res_name, row_num, col_num);

    std::cout << "Output files generated successfully." << std::endl << std::endl;

    return ret;
}

} // namespace file

namespace test
{

void out(const std::string& stand_out_path, const std::string& out_path, int n)
{
    try
    {
        std::vector<float> stand_out = file::read(stand_out_path, n, n, ',');
        std::vector<float> out_vec = file::read(out_path, n, n, ' '); // Assuming space delim for output

        bool passed = true;
        int error_count = 0;

        for (int i = 0; i < n * n; i++)
        {
            // Robust float comparison
            if (std::abs(stand_out[i] - out_vec[i]) > util::EPSILON)
            {
                if (error_count < 10) // Limit error spam
                {
                    std::cout << "Error at index " << i << ": Expected " << stand_out[i]
                              << ", Got " << out_vec[i] << std::endl;
                }
                passed = false;
                error_count++;
            }
        }

        if (passed)
        {
            std::cout << "Correct output - Test PASSED" << std::endl << std::endl;
        }
        else
        {
            std::cout << "Wrong output - Test FAILED (Total errors: " << error_count << ")" << std::endl << std::endl;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Test execution failed: " << e.what() << std::endl;
    }
}

float compute_mean_squared_error(const std::string& original_out_path, const std::string& out_path, int n)
{
    try
    {
        std::vector<float> original_out = file::read(original_out_path, n, n, ',');
        std::vector<float> out_vec = file::read(out_path, n, n, ' ');
        double res = 0; // Use double for accumulation precision

        for (int i = 0; i < n * n; i++)
        {
            res += pow(original_out[i] - out_vec[i], 2);
        }

        return (float)(res / (n * n));
    }
    catch (const std::exception& e)
    {
        std::cerr << "MSE Calculation failed: " << e.what() << std::endl;
        return -1.0f;
    }
}

} // namespace test

#endif // __UTILS_CUH__