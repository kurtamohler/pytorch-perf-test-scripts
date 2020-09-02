#include <iostream>
#include <chrono>
#include <vector>
#include <random>

// This is the original implementation of comptue_numel()
int64_t numel(std::vector<int64_t>& sizes) {
  int64_t n = 1;
  for (auto s : sizes) {
    n *= s;
  }
  return n;
}

// This is an implementation of comptue_numel() that does overflow checking with
// a divide operation. This sould take significantly more time than the orignal.
int64_t numel_overflow_div(std::vector<int64_t>& sizes) {
  int64_t n = 1;
  for (auto s : sizes) {
    if (! (n <= (std::numeric_limits<int64_t>::max() / s)) ){
      std::cout << "int64 overflow while computing numel for ndim = " << sizes.size() << std::endl;
      exit(1);
    }
    n *= s;
  }
  return n;
}

// This is an implementation of comptue_numel() that does overflow checking with
// casting and comparison operations. This sould hopefully not take significantly
// more time than the original.
int64_t numel_overflow_cast(std::vector<int64_t>& sizes) {
  uint64_t n = 1;
  for (uint64_t s : sizes) {
    uint64_t n_next = n * s;
    if (! ((n_next >= n) || (n_next >= s)) ){
      std::cout << "int64 overflow while computing numel for ndim = " << sizes.size() << std::endl;
      exit(1);
    }
    n = n_next;
  }
  if (! (n <= std::numeric_limits<int64_t>::max()) ) {
      std::cout << "int64 overflow while computing numel for ndim = " << sizes.size() << std::endl;
      exit(1);
  }
  return n;
}

std::vector<int64_t> generate_sizes(std::default_random_engine& rng_engine, uint64_t ndim) {
  std::vector<int64_t> sizes(ndim);
  std::uniform_int_distribution<int64_t> distribution(1, 10);
  for (uint64_t dim = 0; dim < ndim; dim++) {
    sizes[dim] = distribution(rng_engine);
  }
  int64_t num = numel(sizes);
  if ((num != numel_overflow_div(sizes)) || (num != numel_overflow_cast(sizes))) {
    std::cout << "numel methods do not agree" << std::endl;
    exit(1);
  }
  return sizes;
}

enum BenchType {
  BENCH_ORIG = 0,
  BENCH_OVERFLOW_DIV,
  BENCH_OVERFLOW_CAST
};

// NOTE: the total_numel arg is used to ensure that the numel*() calls don't get
// optimized out. After calling this function, we should give total_numel to another
// function that can't optimize it out. Below, we use it as a random generator seed.
template<BenchType benchType>
double run_bench(std::vector<std::vector<int64_t>> sizes_to_run, uint64_t ndim, int64_t& total_numel) {
  double time_per_iter = 0;
  for (int warmup = 0; warmup < 2; warmup++) {
    auto start = std::chrono::system_clock::now();
    for (auto sizes : sizes_to_run) {
      if (benchType == BENCH_ORIG) {
        total_numel += numel(sizes);
      } else if (benchType == BENCH_OVERFLOW_DIV) {
        total_numel += numel_overflow_div(sizes);
      } else if (benchType == BENCH_OVERFLOW_CAST) {
        total_numel += numel_overflow_cast(sizes);
      }
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    time_per_iter = elapsed.count() / static_cast<double>(sizes_to_run.size());
  }
  return time_per_iter;
}

int main() {
  std::default_random_engine rng_engine;
  uint64_t timed_iters = 1000000;
  uint64_t max_ndim = 21;
  int64_t total_numel = 0;

  std::vector<double> numel_times(max_ndim);
  std::vector<double> numel_overflow_div_times(max_ndim);
  std::vector<double> numel_overflow_cast_times(max_ndim);

  rng_engine.seed(0);
  for (uint64_t ndim = 1; ndim < max_ndim; ndim++) {
    std::vector<std::vector<int64_t>> sizes_to_run(timed_iters);
    for (uint64_t iter = 0; iter < timed_iters; iter++) {
      sizes_to_run[iter] = generate_sizes(rng_engine, ndim);
    }
    numel_times[ndim] = run_bench<BENCH_ORIG>(sizes_to_run, ndim, total_numel);
    numel_overflow_div_times[ndim] = run_bench<BENCH_OVERFLOW_DIV>(sizes_to_run, ndim, total_numel);
    numel_overflow_cast_times[ndim] = run_bench<BENCH_OVERFLOW_CAST>(sizes_to_run, ndim, total_numel);
  }
  // As mentioned above, this is to ensure the numel*() calls don't get optimized out
  rng_engine.seed(total_numel);

  std::cout << "ndim numel-orig numel-overflow-div numel-overflow-cast" << std::endl;
  for (uint64_t ndim = 1; ndim < max_ndim; ndim++) {
    std::cout << ndim
      << " " << numel_times[ndim]
      << " " << numel_overflow_div_times[ndim]
      << " " << numel_overflow_cast_times[ndim]
      << std::endl;
  }
}
