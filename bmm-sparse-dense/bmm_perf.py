# Compares performance of torch.bmm(sparse, dense) with
# a workaround that was used in the absense of this function
# Since performance of sparse tensor operations is highly
# dependent on the number of nonzero elements, we have to
# run the measurement while varying both tensor sizes
# and percentages of nonzeros.

import torch
import random
import time
import math

torch.set_num_threads(1)

# Create a sparse matrix of specified sparsity and size, with
# randomized nonzero element positions and values
def get_rand_sparse_matrix(dim0, dim1, sparsity, device):
    size = [dim0, dim1]
    matrix = torch.zeros(size, dtype=torch.float, device=device)
    num_elements = dim0 * dim1
    num_nnz = max(int(num_elements * (1-sparsity)), 1)

    nnz_inds_1D = random.sample(range(num_elements), num_nnz)
    for nnz_ind_1D in nnz_inds_1D:
        nnz_ind_2D = (
            int(nnz_ind_1D % dim0),
            int(nnz_ind_1D / dim0)
        )
        rand_num = random.random()
        while rand_num == 0:
            rand_num = random.random()

        matrix[nnz_ind_2D] = rand_num

    return matrix.to_sparse()

def calc_bmm_builtin(a, b):
    ab = a.bmm(b)
    return ab

def calc_bmm_workaround(a, b):
    ab = torch.stack([ai.mm(bi) for ai, bi in zip(a, b)])
    return ab

def check_equal(tensor0, tensor1, epsilon):
    return ((tensor0 - tensor1).abs() <= epsilon).all()

# Measure the average time to calculate BMM on a randomized set of tensors
# using three methods: builtin BMM, builtin BMM given a pre-coalesced sparse tensor,
# and a workaround to calculate BMM without the builtin method
def measure_bmm_time(num_matrices, a_dim0, a_dim1, b_dim1, sparsity, timed_iters, device):
    b_dim0 = a_dim1
    a = []

    for mat_ind in range(num_matrices):
        a.append(get_rand_sparse_matrix(a_dim0, a_dim1, sparsity, device).coalesce())

    a_stacked = torch.stack(a)
    a_stacked_coalesced = torch.stack(a).coalesce()
    b = torch.rand([num_matrices, b_dim0, b_dim1], dtype=torch.float, device=device)

    # do one warmup loop
    for _ in range(2):
        start_workaround = time.time()
        for _ in range(timed_iters):
            ab_workaround = calc_bmm_workaround(a, b)
        time_workaround = (time.time() - start_workaround) / timed_iters

        start_builtin = time.time()
        for _ in range(timed_iters):
            ab_builtin = calc_bmm_builtin(a_stacked, b)
        time_builtin = (time.time() - start_builtin) / timed_iters

        start_builtin_coalesced = time.time()
        for _ in range(timed_iters):
            ab_builtin_coalesced = calc_bmm_builtin(a_stacked_coalesced, b)
        time_builtin_coalesced = (time.time() - start_builtin_coalesced) / timed_iters

        # make sure the calculations match
        # if not ab_workaround.eq(ab_builtin).all():
        # if not ab_workaround.equal(ab_builtin).all():
        if not check_equal(ab_workaround, ab_builtin, 1e-4):
            print("ab_builtin doesn't match ab_workaround!")
            exit(1)
        # # if not ab_workaround.eq(ab_builtin_coalesced).all():
        if not check_equal(ab_workaround, ab_builtin_coalesced, 1e-4):
            print("ab_builtin_coalesced doesn't match ab_workaround!")
            exit(1)

    return time_workaround, time_builtin, time_builtin_coalesced


print("num_matrices squ_mat_elements sparsity workaround builtin builtin-precoalesced")

for device in ['cuda', 'cpu']:
    for num_matrices in [10, 100, 1000, 10_000]:
        for dim_size in [10, 100, 1000]:
            for sparsity in [0.999, 0.99, 0.9, 0]:
                nnz = int(num_matrices*dim_size*dim_size*(1.-sparsity))

                if nnz <= 100_000 and nnz > num_matrices:
                    # The number of iterations to average over is inversely
                    # proportional to the number of elements in the sparse tensor.
                    # This is because a smaller sparse tensor will take less time,
                    # so we'll want to run more iterations to get a more stable
                    # measurement.
                    timed_iters = int(math.ceil(5_000./nnz))
                    time_workaround, time_builtin, time_builtin_coalesced = measure_bmm_time(
                        num_matrices,
                        dim_size, dim_size, dim_size,
                        sparsity,
                        timed_iters,
                        device
                    )
                    print("%s %d %d %f %f %f %f" % (
                        device,
                        num_matrices,
                        dim_size**2,
                        sparsity,
                        time_workaround,
                        time_builtin,
                        time_builtin_coalesced
                    ))
