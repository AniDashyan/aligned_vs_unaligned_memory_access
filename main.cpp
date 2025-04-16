#include <immintrin.h> // AVX  and _mm_malloc
#include <format>
#include <cstdlib>
#include <cstring> 
#include <cmath>       
#include <utility>    
#include "kaizen.h"    

// Ensure AVX support is enabled
#ifndef __AVX__
#error "AVX is required. Compile with -mavx or -mavx2 (e.g., g++ -O3 -mavx2 or clang++ -O3 -mavx2)."
#endif

// Warn about ARM-based macOS (no native AVX)
#ifdef __APPLE__
#ifdef __arm64__
#warning "ARM-based macOS (Apple Silicon) does not natively support AVX. Running via Rosetta 2 or use NEON for native support."
#endif
#endif


double sum_scalar(const double* arr, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += arr[i];
    }
    return sum;
}

double sum_avx(const double* arr, size_t n, bool aligned) {
    __m256d sum_vec = _mm256_setzero_pd(); // Initialize 4 doubles to 0
    size_t i = 0;

    //  process 4 doubles per iteration
    for (; i <= n - 4; i += 4) {
        __m256d data;
        if (aligned) {
            data = _mm256_load_pd(&arr[i]); // Aligned load
        } else {
            data = _mm256_loadu_pd(&arr[i]); // Unaligned load
        }
        sum_vec = _mm256_add_pd(sum_vec, data);
    }

   
    __m256d temp = _mm256_hadd_pd(sum_vec, sum_vec); // [a+b, c+d, a+b, c+d]
    double sum_array[4];
    _mm256_storeu_pd(sum_array, temp);
    double sum = sum_array[0] + sum_array[2]; // (a+b) + (c+d)

    // Handle remaining elements
    for (; i < n; ++i) {
        sum += arr[i];
    }

    return sum;
}

double measure_time(const double* arr, size_t n, bool aligned, double& result, size_t runs) {
    zen::timer t;

    t.start();
    for (size_t i = 0; i < runs; ++i) {
        result = sum_avx(arr, n, aligned);
    }
    t.stop();
    return t.duration<zen::timer::nsec>().count() / runs;
}

bool is_aligned(const void* ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

std::pair<int, int> parse_args(int argc, char** argv) {
    zen::cmd_args args(argv, argc);
    int size = 1'000'000;
    int runs = 1000;
    if (!args.is_present("--size")|| !args.is_present("--runs")) {
        zen::log(zen::color::yellow("No --size or --runs provided. Using default values: "));
        return {size, runs};
    } else {
        size = std::stoi(args.get_options("--size")[0]);
        runs = std::stoi(args.get_options("--runs")[0]);
    }
    return {size, runs};
}


int main(int argc, char** argv) {
    auto [size, runs] = parse_args(argc, argv);

   
    void* temp_aligned = _mm_malloc(size * sizeof(double), 32);
    if (!temp_aligned) {
       zen::log(zen::color::red("Aligned memory allocation failed"));
        return 1;
    }
    double* aligned_array = static_cast<double*>(temp_aligned);
    if (!is_aligned(aligned_array, 32)) {
        zen::print(zen::color::red("Aligned array not 32-byte aligned\n"));
        _mm_free(aligned_array);
        return 1;
    }

    for (size_t i = 0; i < size; ++i) {
        aligned_array[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // Allocate unaligned memory
    void* temp_unaligned = _mm_malloc((size + 1) * sizeof(double), 32);
    if (!temp_unaligned) {
       zen::log(zen::color::red("Unaligned memory allocation failed"));
        _mm_free(aligned_array);
        return 1;
    }

    double* unaligned_raw = static_cast<double*>(temp_unaligned);
    char* unaligned_start = (char*)unaligned_raw + 4;
    double* unaligned_array = (double*)unaligned_start;
    memcpy(unaligned_array, aligned_array, size * sizeof(double));

    
    double scalar_sum = sum_scalar(aligned_array, size);
    double aligned_sum = sum_avx(aligned_array, size, true);
    double unaligned_sum = sum_avx(unaligned_array, size, false);

   
    zen::log(std::format("Running with: size={}, runs={}", size, runs) + "\n");
   
    zen::print(std::format("Scalar sum:    {:.3f}\n", scalar_sum));
    zen::print(std::format("Aligned sum:   {:.3f}\n", aligned_sum));
    zen::print(std::format("Unaligned sum: {:.3f}\n", unaligned_sum));

    if (std::isnan(unaligned_sum)) {
        zen::print(zen::color::red("Error: Unaligned sum is NaN\n"));
        _mm_free(aligned_array);
        _mm_free(unaligned_raw);
        return 1;
    }

    double aligned_result, unaligned_result;
    double aligned_time = measure_time(aligned_array, size, true, aligned_result, runs);
    double unaligned_time = measure_time(unaligned_array, size, false, unaligned_result, runs);

    zen::print(std::format("\nPerformance (average over {} runs):\n", runs));
    zen::print(std::format("Aligned AVX time:   {:.3f} ms\n", aligned_time));
    zen::print(std::format("Unaligned AVX time: {:.3f} ms\n", unaligned_time));
    zen::print(std::format("Performance ratio (unaligned/aligned): {:.3f}\n", 
                           unaligned_time / aligned_time));

    // Free memory
    _mm_free(aligned_array);
    _mm_free(unaligned_raw);

    return 0;
}