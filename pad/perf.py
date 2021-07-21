import torch
from torch.utils import benchmark
import numpy as np
from itertools import product

devices = [
    'cpu',
    'cuda'
]

dtypes = [
    torch.float,
    torch.double
]

cases = [
    # input size, pad_width, constant_values
    ((10,), 10, None),
    ((100,), 100, None),
    ((1000,), 1000, None),
    ((10000,), 10000, None),
    ((10000,), 10, None),
    ((10, 10, 10), 10, None),
    ((10, 10, 10), ((1000,), (0,), (0,)), None),
    ((20, 10, 10), 10, None),
    ((30, 10, 10), 10, None),
    ((100, 10, 10), 10, None),
    ((10, 10, 10), 10, ((0, 1), (2, 3), (4, 5))),
    ((10, 10, 10), ((10, 10), (10, 10), (10, 10)), ((0, 1), (2, 3), (4, 5))),
    ((100, 100, 100), ((10, 10), (10, 10), (10, 10)), None),
]

num_iters = 100

print('device dtype case_idx time_torch time_numpy torch_speedup')
print()

for device, dtype, (case_idx, (input_size, pad_width, constant_values)) in product(devices, dtypes, enumerate(cases)):
    time_numpy = benchmark.Timer(
        setup=f'import numpy as np; a = torch.randn({input_size}, dtype={dtype}).numpy()',
        stmt=f'np.pad(a, {pad_width}, constant_values={constant_values})'
    ).timeit(num_iters).mean

    time_torch = benchmark.Timer(
        setup=f'a = torch.randn({input_size}, dtype={dtype}, device="{device}")',
        stmt=f'torch.pad(a, {pad_width}, constant_values={constant_values})'
    ).timeit(num_iters).mean

    torch_speedup = time_numpy / time_torch


    print(f'{device} {dtype} {case_idx} {time_torch:.2e} {time_numpy:.2e} {torch_speedup:.2f}')
    if case_idx == (len(cases) - 1):
        print()


