import torch
import sys
from time import time
from itertools import product
from math import prod
from random import sample, seed


torch.manual_seed(0)
seed(0)


test_cases = product(
    # device
    sys.argv[1:],

    # dtype
    [torch.float32, torch.float64],

    # input size
    [
        (2, 10, 10), (2, 100, 100), (2, 200, 200),
        (10, 10, 10), (10, 100, 100), (10, 200, 200),
        (100, 10, 10), (100, 100, 100), (100, 200, 200),
        (2, 2, 10, 10), (2, 2, 100, 100), (2, 2, 200, 200),
        (20, 2, 10, 10), (20, 2, 100, 100), (20, 2, 200, 200),
        (2, 20, 10, 10), (2, 20, 100, 100), (2, 20, 200, 200),
    ],

    # ratio of singular matrices
    [0., 0.5, 1.],
)

min_total_time = 0.1
sample_count = 5

def measure_run_time(input, min_total_time, include_backward=False):
    need_sync = (input.device.type == 'cuda')

    timed_iters = 2

    run_time = None

    # Create zeros tensor to use in backwards
    with torch.no_grad():
        det = torch.linalg.det(input)
        zeros_backward = torch.zeros_like(det)
    
    while run_time is None:
        if include_backward:
            for warmup in range(2):
                start = time()

                for iter in range(timed_iters):
                    det = torch.linalg.det(input)
                    det.backward(zeros_backward)

                if need_sync:
                    torch.cuda.synchronize(input.device)

                end = time()
        else:
            for warmup in range(2):
                start = time()

                for iter in range(timed_iters):
                    det = torch.linalg.det(input)

                if need_sync:
                    torch.cuda.synchronize(input.device)

                end = time()

        total_time = end - start

        # If the total time was long enough, calculate the time per iteration.
        # Otherwise, increase the iteration count appropriately to try to reach
        # the target time.
        if total_time >= min_total_time:
            run_time = total_time / timed_iters
        else:
            timed_iters = int(timed_iters * 1.2 * min_total_time / total_time)

    assert total_time >= min_total_time

    return run_time

print('| device | dtype | input_size | singular_ratio | forward time | backward time |')
print(('| --- ' * 6) + '|')

def gen_input(input_size, singular_ratio, device, dtype):
    assert singular_ratio >= 0 and singular_ratio <= 1

    matrix_size = input_size[-2:]
    batch_size = input_size[:-2]
    num_matrices = prod(batch_size)
    num_singular_matrices_f = singular_ratio * num_matrices
    num_singular_matrices = round(num_singular_matrices_f)

    if abs(num_singular_matrices_f - num_singular_matrices) > 0.05:
        raise RuntimeError(
            f'singular_ratio * num_matrices ({singular_ratio} * {num_matrices}) '
            f'= {num_singular_matrices_f} is not close to a whole number')

    if num_singular_matrices == 0:
        return torch.randn(input_size, device=device, dtype=dtype, requires_grad=True)
    elif num_singular_matrices == num_matrices:
        return torch.ones(input_size, device=device, dtype=dtype, requires_grad=True)

    is_singular = [False] * num_matrices
    
    # Randomly pick which matrices are singular
    for matrix_idx in sample(range(num_matrices), num_singular_matrices):
        is_singular[matrix_idx] = True

    input = torch.empty(input_size, device=device, dtype=dtype)

    view_size = (num_matrices,) + matrix_size
    input_view = input.view(view_size)

    for matrix_idx in range(num_matrices):
        if is_singular[matrix_idx]:
            input_view[matrix_idx] = torch.ones(matrix_size, device=device, dtype=dtype)
        else:
            input_view[matrix_idx] = torch.randn(matrix_size, device=device, dtype=dtype)

    input.requires_grad_(True)
    return input

for device, dtype, input_size, singular_ratio in test_cases:
    run_times_forward = []
    run_times_both = []
    run_times_backward = []
    for sample_num in range(sample_count):
        input = gen_input(input_size, singular_ratio, device, dtype)

        run_time_forward = measure_run_time(input, min_total_time, include_backward=False)
        run_time_both = measure_run_time(input, min_total_time, include_backward=True)
        run_time_backward = run_time_both - run_time_forward

        run_times_forward.append(run_time_forward)
        #run_times_both.append(run_time_both)
        run_times_backward.append(run_time_backward)

    median_forward = torch.tensor(run_times_forward, dtype=torch.float64).median()
    #median_both = torch.tensor(run_times_both, dtype=torch.float64).median()
    median_backward = torch.tensor(run_times_backward, dtype=torch.float64).median()

    print(f'| {device} | {dtype} | {input_size} | {singular_ratio} | {median_forward} | {median_backward} |')
