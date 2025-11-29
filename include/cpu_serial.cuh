#ifndef __FILTERING_CUH__
#define __FILTERING_CUH__

#include <utils.cuh>

namespace cpu
{

float filter_pixel(float * image,
                   float * weights,
                   int n,
                   int patch_size,
                   int pixel_row,
                   int pixel_col,
                   float sigma)
{
    float res = 0;
    float sum_w = 0; // sum_w is the Z(i) of w(i, j) formula
    float dist;
    float w;
    int patch_row_start = pixel_row - patch_size / 2;
    int patch_col_start = pixel_col - patch_size / 2;

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
    std::vector<float> weights = util::compute_inside_weights(patch_size, patch_sigma);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res[i * n + j] = filter_pixel(image, weights.data(), n, patch_size, i, j, filter_sigma);
        }
    }

    return res;
}

} // namespace cpu

#endif // __FILTERING_CUH__