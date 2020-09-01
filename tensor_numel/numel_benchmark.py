# This script benchmarks `torch.empty` given size arrays of different
# numbers of dimensions. The purpose is to find out if adding multiply
# overflow detection to c10::TensorImplt::compute_numel() will significantly
# affect performance


import torch
import numpy
import time

# Set the numpy random seed to minimize run-to-run noise
numpy.random.seed(0)

# Generate a randomized size array of specified number of dimensions.
# The product of the array will be between num_numel and max_numel.
def generate_sizes_set(ndim, min_numel, max_numel):
    numel = 0
    while numel > max_numel or numel < min_numel:
        sizes_set = numpy.random.randint(1, 6, ndim)
        numel = sizes_set.prod()
    return sizes_set, numel

max_num_dims = 22
sizes_set, numel = generate_sizes_set(max_num_dims, 0x3_f000_0000, 0x4_0000_0000)

print(f'set of sizes: {sizes_set}')
print('numel: 0x%x' % sizes_set.prod())

num_iters = 10000

print('num_dims time_per_empty_call')
for num_dims in range(1, max_num_dims + 1):
    sizes_to_check = []
    for i in range(num_iters):
        size = sizes_set.copy()
        numpy.random.shuffle(size)
        # Randomly combine pairs of `size` elements until we have the correct number
        # of dimensions
        while len(size) != num_dims:
            idx_to_reduce = numpy.random.randint(0, len(size))
            num_to_reduce = size[idx_to_reduce]
            size = numpy.delete(size, idx_to_reduce)
            idx_to_mul = numpy.random.randint(0, len(size))
            size[idx_to_mul] *= num_to_reduce

        # We want every size array that we use in the benchmark to have the same
        # product so that the resulting numel of the `torch.empty` calls are all
        # the same, minimizing any differences in memory allocation times 
        assert size.prod() == numel
        sizes_to_check.append(size)
    
    start = time.time()
    for size  in sizes_to_check:
        torch.empty(*size, dtype=torch.int8)
    end = time.time()

    time_per_iter = (end - start) / num_iters
    print(f'{len(size)} {time_per_iter}')