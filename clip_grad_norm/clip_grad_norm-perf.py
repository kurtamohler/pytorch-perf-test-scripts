import torch
import sys
from time import time
from itertools import product

def measure_run_time(parameters, max_norm, norm_type, min_total_time):
    need_sync = (parameters[0].device.type == 'cuda')

    timed_iters = 2

    run_time = None
    
    while run_time is None:
        for warmup in range(2):
            start = time()

            for iter in range(timed_iters):
                torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)

            if need_sync:
                torch.cuda.synchronize(parameters[0].device)

            end = time()

        total_time = end - start

        # If the total time was long enough, calculate the time per iteration.
        # Otherwise, increase the iteration count appropriately to try to reach
        # the target time.
        if total_time >= min_total_time:
            run_time = total_time / timed_iters
        else:
            timed_iters = int(timed_iters * 1.2 * min_total_time / total_time)

    return run_time, timed_iters

test_cases = product(
    # device
    sys.argv[1:],

    # dtype
    [torch.float32, torch.float64],

    # Number of parameters
    [1, 2, 10, 100],

    # Parameter size
    [(10,), (1000,), (100_000,), (1_000_000,)],

    # norm_type
    [float('inf')]
)

min_total_time = 0.1
sample_count = 5
mul_factor = 10
max_norm = 0.1

print('| device | dtype | num_params | param_size | norm_type | run time |')
print(('| --- ' * 6) + '|')

torch.manual_seed(0)

for device, dtype, num_params, param_size, norm_type in test_cases:
    run_times = []
    for sample_num in range(sample_count):
        try:
            parameters = [torch.randn(param_size, device=device, dtype=dtype, requires_grad=True) for _ in range(num_params)]
        except RuntimeError:
            # Some of the input sizes are too big to allocate on some devices.
            # If that happens, just skip this size
            break

        for param in parameters:
            param.mul(mul_factor).sum().backward()

        run_time, start_timed_iters = measure_run_time(parameters, max_norm, norm_type, min_total_time)
        run_times.append(run_time)

    median_time = torch.tensor(run_times, dtype=torch.float64).median()

    print(f'| {device} | {dtype} | {num_params} | {param_size} | {norm_type} | {median_time} |')
