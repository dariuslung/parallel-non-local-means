#include <iostream>
#include <vector>
#include <cpu_serial.cuh>
#include <gpu_global_mem.cuh>
#include <gpu_shared_mem.cuh>
#include <cstdlib>
#include <string> 
#include <getopt.h>

// Function to display usage instructions
void print_usage(const char* program_name)
{
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  -n, --image-num <int>      Set image number (Default: 0)\n"
              << "  -p, --patch-size <int>     Set patch size (Default: 5)\n"
              << "  -f, --filter-sigma <float> Set filter sigma (Default: 0.06)\n"
              << "  -s, --patch-sigma <float>  Set patch sigma (Default: 0.8)\n"
              << "  -g, --use-gpu <0/1>        Enable/Disable GPU (Default: 0)\n"
              << "  -m, --use-shared <0/1>     Enable/Disable Shared Mem (Default: 0)\n"
              << "  -h, --help                 Show this help message\n";
}

int main(int argc, char** argv)
{   
    std::cout << std::endl;
    util::Timer timer(true);
    
    // --- Parameters ---
    int n = 0;
    int image_num = 0;
    int patch_size = 5;
    float filter_sigma = 0.06f;
    float patch_sigma = 0.8f;
    bool use_gpu = false;
    bool use_shared_mem = false;
    std::string image_path = "./data/in/noisy_house.txt";

    // --- Define long options ---
    const struct option long_options[] =
    {
        {"image-num",    required_argument, nullptr, 'n'},
        {"patch-size",   required_argument, nullptr, 'p'},
        {"filter-sigma", required_argument, nullptr, 'f'},
        {"patch-sigma",  required_argument, nullptr, 's'},
        {"use-gpu",      required_argument, nullptr, 'g'},
        {"use-shared",   required_argument, nullptr, 'm'},
        {"help",         no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int option_index = 0;
    int c;

    // --- Parse options ---
    while ((c = getopt_long(argc, argv, "n:p:f:s:g:m:h", long_options, &option_index)) != -1)
    {
        switch (c)
        {
            case 'n':
                image_num = std::atoi(optarg);
                break;
                
            case 'p':
                patch_size = std::atoi(optarg);
                break;

            case 'f':
                filter_sigma = std::atof(optarg);
                break;

            case 's':
                patch_sigma = std::atof(optarg);
                break;

            case 'g':
                use_gpu = (std::atoi(optarg) != 0);
                break;

            case 'm':
                use_shared_mem = (std::atoi(optarg) != 0);
                break;

            case 'h':
                print_usage(argv[0]);
                return 0;

            case '?':
                // Invalid option
                print_usage(argv[0]);
                return 1;

            default:
                return 1;
        }
    }

    // --- Output configuration for verification ---
    std::cout << "Configuration Loaded:\n"
              << "  Patch Size:   " << patch_size << "\n"
              << "  Filter Sigma: " << filter_sigma << "\n"
              << "  Patch Sigma:  " << patch_sigma << "\n"
              << "  Use GPU:      " << (use_gpu ? "True" : "False") << "\n"
              << "  Use Shared:   " << (use_shared_mem ? "True" : "False") << "\n";
    
    // --- Set image size and path based on image number ---
    if (image_num == 0) {
        n = 64;
        image_path = "./data/in/noisy_house.txt";
    }

    else if (image_num == 1) {
        n = 128;
        image_path = "./data/in/noisy_flower.txt";
    }

    else if (image_num == 2) {
        n = 256;
        image_path = "./data/in/noisy_lena.txt";
    }
    std::cout << "  Image path: " << image_path << " (" << n << "x" << n << ")" << std::endl << std::endl;

    // --- Read files ---
    std::vector<float> image = file::read(image_path, n, n, ',');
    std::cout << "Image loaded." << std::endl;
    std::vector<float> filtered_image;

    // --- CPU Image Filtering ---
    if (!use_gpu) {
        timer.start("CPU filtering");
        filtered_image = cpu::filter_image(image.data(), n, patch_size, patch_sigma, filter_sigma);
        timer.stop();
    }

    // --- GPU Image Filtering ---
    if (use_gpu) {
        if (!use_shared_mem) {
            timer.start("GPU filtering (global memory)");
            filtered_image = gpu_global_mem::filter_image(image.data(), n, patch_size, patch_sigma, filter_sigma);
            timer.stop();
        }
        else {
            timer.start("GPU filtering (shared memory)");
            filtered_image = gpu_shared_mem::filter_image(image.data(), n, patch_size, patch_sigma, filter_sigma);
            timer.stop();
        }
    }

    // --- Print parameters ---
    prt::parameters(patch_size, filter_sigma, patch_sigma);

    // --- Residual computation ---
    std::vector<float> residual = util::compute_residual(image, filtered_image, n);

    // --- File writing ---
    std::string outPath = file::write_images(filtered_image, residual, patch_size, filter_sigma, patch_sigma , n, n, use_gpu);

    std::cout << std::endl;
    return 0;
}