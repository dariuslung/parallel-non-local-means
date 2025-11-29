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

extern __shared__ float s[];

namespace util
{

// --- Timer ---
class Timer
{
public:
    Timer(bool print) : print(print) {}

    void start(std::string operation_desc)
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
            std::cout << "\n" << _operation_desc << " time: " << duration / 1e3 << "ms\n" << std::endl;
        }
    }

private:
    float duration;
    bool print;
    std::string _operation_desc;
    std::chrono::high_resolution_clock::time_point t1, t2;
};

// --- Host & Device utils ---

__host__ __device__ bool is_in_bounds(int n, int x, int y)
{
    return x >= 0 && x < n && y >= 0 && y < n;
}

__host__ __device__ float compute_weight(float dist, float sigma) // compute weight without "/z(i)" division
{
    return expf(-dist / (sigma * sigma));
}

// patch-to-patch euclidean distance
__host__ __device__ float compute_patch_distance(float * image,
                                                 float * weights,
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
            if (is_in_bounds(n, p1_row_start + i, p1_col_start + j) && is_in_bounds(n, p2_row_start + i, p2_col_start + j))
            {
                temp = image[(p1_row_start + i) * n + p1_col_start + j] - image[(p2_row_start + i) * n + p2_col_start + j];
                ans += weights[i * patch_size + j] * temp * temp;
            }
        }
    }

    return ans;
}

// --- Host utils ---

float * compute_inside_weights(int patch_size, float patch_sigma)
{
    float * weights = new float[patch_size * patch_size];
    int central_pixel_row = patch_size / 2;
    int central_pixel_col = central_pixel_row;
    float dist;
    float sum_w = 0;

    for (int i = 0; i < patch_size; i++)
    {
        for (int j = 0; j < patch_size; j++)
        {
            dist = (central_pixel_row - i) * (central_pixel_row - i) +
                   (central_pixel_col - j) * (central_pixel_col - j);
            weights[i * patch_size + j] = exp(-dist / (2 * (patch_sigma * patch_sigma)));
            sum_w += weights[i * patch_size + j];
        }
    }

    for (int i = 0; i < patch_size; i++)
    {
        for (int j = 0; j < patch_size; j++)
        {
            weights[i * patch_size + j] = weights[i * patch_size + j] / sum_w;
        }
    }

    return weights;
}

std::vector<float> compute_residual(std::vector<float> image, std::vector<float> filtered_image, int n)
{
    std::vector<float> res(n * n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res[i * n + j] = image[i * n + j] - filtered_image[i * n + j];
        }
    }
    std::cout << "Residual calculated" << std::endl << std::endl;

    return res;
}

// --- Device utils ---

// patch-to-patch euclidean distance
__device__ float cuda_compute_patch_distance(float * image,
                                             float * weights,
                                             int n,
                                             int patch_size,
                                             int p1_row_start,
                                             int p1_col_start,
                                             int p2_row_start,
                                             int p2_col_start)
{
    float *patches = s;

    float ans = 0;
    float temp;

    for (int i = 0; i < patch_size; i++)
    {
        for (int j = 0; j < patch_size; j++)
        {
            if (is_in_bounds(n, p1_row_start + i, p1_col_start + j) && is_in_bounds(n, p2_row_start + i, p2_col_start + j))
            {
                // Note: accessing shared memory 'patches' here based on original logic
                temp = patches[i * n + p1_col_start + j] -
                       image[(p2_row_start + i) * n + p2_col_start + j];
                ans += weights[i * patch_size + j] * temp * temp;
            }
        }
    }

    return ans;
}

} // --- Namespace utils ---

namespace prt
{

void row_major_array(float * arr, int n, int m)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            std::cout << arr[i * m + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void row_major_vector(std::vector<float> vector, int n, int m)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            std::cout << vector[i * m + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void parameters(int patch_size, float filter_sigma, float patch_sigma)
{
    std::cout << "Image filtered: "   << std::endl
              << "-Patch size "       << patch_size   << std::endl
              << "-Patch sigma "      << patch_sigma  << std::endl
              << "-Filter Sigma "     << filter_sigma << std::endl << std::endl;
}

} // --- Namespace prt ---

namespace file
{

std::vector<float> read(std::string file_path, int n, int m, char delim)
{
    std::vector<float > image(n * m);
    std::ifstream myfile(file_path);
    std::ifstream input(file_path);
    std::string s;

    for (int i = 0; i < n; i++)
    {
        std::getline(input, s);
        std::istringstream iss(s);
        std::string num;
        int j = 0;
        while (std::getline(iss, num, delim))
        {
            image[i * m + j++] = std::stof(num);
        }
    }

    return image;
}

std::string write(std::vector<float> image, std::string file_name, int row_num, int col_num)
{
    std::vector<std::string> out;

    for (int i = 0; i < row_num; i++)
    {
        for (int j = 0; j < col_num; j++)
        {
            out.push_back(std::to_string(image[i * col_num + j]) + " ");
        }
        out.push_back("\n");
    }

    std::ofstream output_file("./data/out/" + file_name + ".txt");
    std::ostream_iterator<std::string> output_iterator(output_file, "");
    std::copy(out.begin(), out.end(), output_iterator);

    return "./data/out/" + file_name + ".txt";
}

std::string write_images(std::vector<float> filtered_image,
                         std::vector<float > residual,
                         int patch_size,
                         float filter_sigma,
                         float patch_sigma,
                         int row_num,
                         int col_num,
                         bool use_gpu)
{
    std::string ret;
    std::string params = std::to_string(patch_size) + "_" +
                         std::to_string(filter_sigma) + "_" +
                         std::to_string(patch_sigma);

    std::string filtered_name = "filtered_image_" + params;
    if (use_gpu)
    {
        filtered_name = "cuda_" + filtered_name;
    }
    ret = file::write(filtered_image, filtered_name, row_num, col_num);

    std::string res_name = "residual_" + params;
    if (use_gpu)
    {
        res_name = "cuda_" + res_name;
    }
    file::write(residual, res_name, row_num, col_num);

    std::cout << "Filtered image written" << std::endl << std::endl;
    std::cout << "Residual written" << std::endl << std::endl;

    return ret;
}

} // --- Namespace file ---

namespace test
{

void out(std::string stand_out_path, std::string out_path, int n)
{
    std::vector<float> stand_out = file::read(stand_out_path, n, n, ',');
    std::vector<float> out_vec = file::read(out_path, n, n, ',');

    for (int i = 0; i < n * n; i++)
    {
        if (stand_out[i] != out_vec[i])
        {
            std::cout << "Error:\t" << stand_out[i] << "\t" << out_vec[i] << std::endl;
        }
    }

    if (stand_out == out_vec)
        std::cout << "Correct output - Test passed" << std::endl << std::endl;
    else
        std::cout << "Wrong output - Test failed" << std::endl << std::endl;
}

float compute_mean_squared_error(std::string original_out_path, std::string out_path, int n)
{
    std::vector<float> original_out = file::read(original_out_path, n, n, ',');
    std::vector<float> out_vec = file::read(out_path, n, n, ',');
    float res = 0;

    for (int i = 0; i < n * n; i ++)
    {
        res += pow(original_out[i] - out_vec[i], 2);
    }

    res = res / (n * n);

    return res;
}

} // namespace test

#endif // __UTILS_CUH__