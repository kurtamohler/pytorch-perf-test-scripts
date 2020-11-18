import torch
import time
import random

torch.set_num_threads(1)


def measure_perf(start, end, step, dtype, device, num_iters):
    for warmup in range(2):
        start_time = time.time()
        for i in range(num_iters):
            torch.arange(start, end, step, dtype=dtype, device=device)
        end_time = time.time()

    return (end_time - start_time) / num_iters


num_iters = 100

print('device dtype num_steps time')
for device in ['cpu']:
    for dtype in [torch.double]:
        for num_steps in [10_000, 100_000, 1_000_000, 2_500_000, 5_000_000, 7_500_000, 10_000_000]:
            start = random.random() * 2 - 1
            step = random.random() * 2 - 1
            end = start + num_steps * step
            total_time = measure_perf(start, end, step, dtype, device, num_iters)

            print(f'{device} {dtype} {num_steps} {total_time}')

