import torch
from torch.utils import benchmark
import numpy as np
from itertools import product

torch.set_num_threads(1)

devices = [
    'cpu',
    'cuda'
]

dtypes = [
    torch.float,
    torch.double
]

####################################
# Compare torch.pad with numpy.pad #
####################################

def bench1():
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
        ((10, 10, 10), 10, 10.0),
        ((10, 10, 10), ((10, 10), (10, 10), (10, 10)), 123),
        ((100, 100, 100), ((10, 10), (10, 10), (10, 10)), None),
    ]

    num_iters = 10000

    print('====================================')
    print('compare with torch.pad')
    print()
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


##################################################
# Compare torch.pad with torch.nn.functional.pad #
##################################################

def bench2():
    cases = [
        # input size, pad_width_new, pad_width_old, constant_values_new, constant_values_old
        ((10,), 10, (10, 10), None, None),
        ((100,), 100, (100, 100), None, None),
        ((1000,), 1000, (1000, 1000), None, None),
        ((10000,), 10000, (10000, 10000), None, None),
        ((10000,), 10, (10, 10), None, None),
        ((10, 10, 10), 10, (10, 10, 10, 10, 10, 10), None, None),
        ((10, 10, 10), ((1000,), (0,), (0,)), (1000, 1000, 0, 0, 0, 0), None, None),
        ((20, 10, 10), 10, (10, 10), None, None),
        ((30, 10, 10), 10, (10, 10), None, None),
        ((100, 10, 10), 10, (10, 10), None, None),
    ]

    num_iters = 10000


    print('====================================')
    print('compare with torch.nn.functional.pad')
    print()
    print('device dtype case_idx time_new time_old new_speedup')
    print()

    for device, dtype, (case_idx, (input_size, pad_width_new, pad_width_old, constant_values_new, constant_values_old)) in product(devices, dtypes, enumerate(cases)):
        time_old = benchmark.Timer(
            setup=f'a = torch.randn({input_size}, dtype={dtype}, device="{device}")',
            stmt=f'torch.nn.functional.pad(a, {pad_width_old}, value={0 if constant_values_old is None else constant_values_old})'
        ).timeit(num_iters).mean

        time_new = benchmark.Timer(
            setup=f'a = torch.randn({input_size}, dtype={dtype}, device="{device}")',
            stmt=f'torch.pad(a, {pad_width_new}, constant_values={constant_values_new})'
        ).timeit(num_iters).mean

        new_speedup = time_old / time_new


        print(f'{device} {dtype} {case_idx} {time_new:.2e} {time_old:.2e} {new_speedup:.2f}')
        if case_idx == (len(cases) - 1):
            print()

bench1()
bench2()
