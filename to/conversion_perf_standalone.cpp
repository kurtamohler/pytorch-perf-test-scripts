#include <iostream>
#include <cstdlib>
#include <chrono>

int main() {
    size_t a_size = 1000000;
    long iters = 10000;
    int* a_32bit = new int[a_size];
    long* a_64bit = new long[a_size];

    // random 32-bit array
    for (size_t i = 0; i < a_size; i++) {
        a_32bit[i] = rand();
    }

    double total_time;

    for (int warmup_iter = 0; warmup_iter < 2; warmup_iter++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < iters; iter++) {
            // Unrolled by 10
            for (size_t i = 0; i < a_size; i+=10) {
                a_64bit[i] = a_32bit[i];
                a_64bit[i+1] = a_32bit[i+1];
                a_64bit[i+2] = a_32bit[i+2];
                a_64bit[i+3] = a_32bit[i+3];
                a_64bit[i+4] = a_32bit[i+4];
                a_64bit[i+5] = a_32bit[i+5];
                a_64bit[i+6] = a_32bit[i+6];
                a_64bit[i+7] = a_32bit[i+7];
                a_64bit[i+8] = a_32bit[i+8];
                a_64bit[i+9] = a_32bit[i+9];
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto time = end_time - start_time;
        total_time = time.count();
    }

    std::cout << "took "
        << total_time/(double)(a_size * iters)
        << " ns per conversion" << std::endl;

    delete [] a_64bit;
    delete [] a_32bit;
}