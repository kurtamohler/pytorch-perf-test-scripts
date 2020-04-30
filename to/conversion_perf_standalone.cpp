#include <iostream>
#include <cstdlib>
#include <chrono>

int main() {
    size_t a_size = 1000000;
    long iters = 10000;
    int* a_32bit = new int[a_size];
    long* a_64bit = new long[a_size];


    // First, random 32-bit array
    for (size_t i = 0; i < a_size; i++) {
        a_32bit[i] = rand();
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; iter++) {
        for (size_t i = 0; i < a_size; i++) {
            a_64bit[i] = a_32bit[i];
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;

    std::cout << "took "
        << time.count()/(double)(a_size * iters)
        << " ns per conversion" << std::endl;

    // std::cout << "took " << time/std::chrono::seconds(1) << " s total" << std::endl;

    delete [] a_64bit;
    delete [] a_32bit;
}