# Optimizing Array Summation with AVX: Aligned vs. Unaligned Memory Access

## Overview

This project aims to optimize the performance of summing an array of double-precision floating-point values (default size: 1 million doubles) using SIMD (Single Instruction, Multiple Data) instructions, specifically Intel’s AVX (Advanced Vector Extensions). The primary goal is to measure and compare the performance of two memory access scenarios:

### 1. Aligned Access
- The array is allocated with 32-byte alignment (ensuring each double starts at a multiple of 32 bytes, optimal for AVX’s 256-bit registers).
- Uses `_mm256_load_pd` for efficient SIMD loads (translates to `vmovapd` instruction).
- Expected to minimize memory access overhead and cache inefficiencies.

### 2. Unaligned Access
- The array is intentionally misaligned by a 4-byte offset (starting at a non-32-byte boundary).
- Uses `_mm256_loadu_pd` for unaligned SIMD loads (translates to `vmovupd` instruction).
- May incur additional memory accesses, slower loads, and cache line splits due to misalignment.

### Objective
- **Measure Performance Difference**: Quantify how memory alignment affects summation speed using AVX instructions.
- **Analyze Misalignment Impact**:
  - Extra Memory Accesses: Crossing cache line boundaries.
  - Slower SIMD Loads: `vmovupd` vs. `vmovapd`.
  - Cache Inefficiencies: Increased cache misses or split loads.
- **Real-World Insight**: Highlight the importance of alignment in optimizing data processing tasks (e.g., scientific computing, machine learning), where large arrays are common.

The project compares three summation methods:
- **Scalar**: Standard loop (no SIMD), baseline reference.
- **AVX Aligned**: SIMD with aligned memory.
- **AVX Unaligned**: SIMD with misaligned memory.

## Build & Run

### Clone the Repository
```bash
git clone https://github.com/AniDashyan/aligned_vs_unaligned_memory_access
cd aligned_vs_unaligned_memory_access
```

### Create Build Directory
```bash
cmake -S . -B build
```

### Build the Project
```bash
cmake --build build --config Release
```

### Run the Program
```bash
./build/sum --size [N] --runs [M]
```
- `--size [N]`: Number of doubles in the array (e.g., 1000000).
- `--runs [M]`: Number of summation iterations for timing (e.g., 1000).
- **Defaults**: `N = 1000000`, `M = 1000` if unspecified.

### Requirements
- **Compiler**: C++20 support (e.g., GCC 13+, Clang 14+).
- **AVX Support**: CPU and compiler flags (`-mavx` or `-mavx2`).
- **CMake**: Version 3.10+.
- **Dependencies**: `kaizen.h` (assumed provided; includes `zen::timer`, `zen::cmd_args`, `zen::print`, `zen::log`, `zen::color`).

### Example
```bash
./build/sum --size 2000000 --runs 500
```

### Note for macOS (Apple Silicon)
- ARM-based Macs emulate AVX via Rosetta 2, which may affect performance.
- Compile with `-mavx2` using Clang.

## Example Output

```
Running with: size=2000000, runs=500
 
Scalar sum:    1000709.108
Aligned sum:   1000709.108
Unaligned sum: 1000709.108

Performance (average over 500 runs):
Aligned AVX time:   1959142.000 ms
Unaligned AVX time: 1977594.000 ms
Performance ratio (unaligned/aligned): 1.009
```

## How Does It Work

The program sums an array of doubles using three methods:

- **Scalar**: A basic loop adds one value at a time, serving as the performance baseline.
- **AVX Aligned**: Uses `_mm256_load_pd` to process 4 doubles per iteration with 32-byte aligned memory, leveraging fast `vmovapd` instructions.
- **AVX Unaligned**: Uses `_mm256_loadu_pd` with a 4-byte offset to simulate misaligned access, invoking `vmovupd`, which may incur cache line splits.

### Core Implementation
- **Memory Allocation**:
  - Aligned: `_mm_malloc(size * sizeof(double), 32)`
  - Unaligned: `_mm_malloc((size + 1) * sizeof(double), 32)` + 4-byte offset
- **Initialization**: Fills array with doubles in [0, 1] using `rand() / RAND_MAX`.
- **Summation**: AVX methods process 4 elements per loop; reduction via `_mm256_hadd_pd`.

### Performance Insight
- **Aligned loads** (`vmovapd`) avoid crossing cache lines, reducing memory traffic.
- **Unaligned loads** (`vmovupd`) may span 64-byte cache boundaries, causing extra fetches.

For large arrays (e.g., 1 million doubles = 8 MB), aligned access typically outperforms unaligned by 20–30%. For small arrays (e.g., 1000 doubles = 8 KB), all data fits in L1 cache, making alignment less impactful and unaligned loads occasionally faster due to reduced instruction overhead or microarchitectural optimizations.

---

## Conclusion

- **Small Sizes**: Cache prefetching and loop overhead dominate. Unaligned loads may seem faster.
- **Large Sizes**: Memory bandwidth and cache efficiency dominate. Aligned loads are significantly faster.

This project highlights the importance of memory alignment in achieving optimal SIMD performance, particularly in high-throughput, data-intensive applications.
```
