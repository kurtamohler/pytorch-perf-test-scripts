# NOTE: Much of this script is based on the performance measurement code in the
# `torch.compile` tutorial:
# https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

import torch
import torch._dynamo
import numpy as np
from itertools import product
import math

torch.set_num_threads(1)

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

def f0(input, padding):
    return torch.ops.aten.replication_pad2d.default(input, padding)

f0_opt = torch.compile(f0)

def measure_perf(input_size, padding, warmup_iters, perf_iters, device, dtype):
    def gen_input():
        return torch.arange(math.prod(input_size), device=device, dtype=dtype).reshape(input_size)

    torch._dynamo.reset()

    for _ in range(warmup_iters):
        a = gen_input()
        with torch.no_grad():
            timed(lambda: f0(a, padding))

    for _ in range(warmup_iters):
        a = gen_input()
        with torch.no_grad():
            timed(lambda: f0_opt(a, padding))

    eager_times = []
    for i in range(perf_iters):
        a = gen_input()
        with torch.no_grad():
            _, eager_time = timed(lambda: f0(a, padding))
        eager_times.append(eager_time)

    compile_times = []
    for i in range(perf_iters):
        a = gen_input()
        with torch.no_grad():
            _, compile_time = timed(lambda: f0_opt(a, padding))
        compile_times.append(compile_time)

    eager_time = np.median(eager_times)
    compile_time = np.median(compile_times)
    speedup = eager_time / compile_time

    return eager_time, compile_time, speedup

test_cases = [
    # size, padding
    ((10, 10, 10), (2, 2, 2, 2)),
    ((1000, 10, 10), (2, 2, 2, 2)),
    ((10, 100, 100), (10, 10, 10, 10)),
    ((10, 100, 100), (100, 100, 100, 100)),
    ((100, 100, 100), (10, 10, 10, 10)),
    ((10, 1000, 1000), (10, 10, 10, 10)),
    ((10, 1000, 1000), (100, 100, 100, 100)),
]

devices = [
    'cpu',
    'cuda',
]

dtypes = [
    torch.float32,
    torch.float64,
    torch.int32,
]

print('input_size, padding, device, dtype, eager_time, compile_time, speedup')

for (input_size, padding), device, dtype in product(test_cases, devices, dtypes):
    eager_time, compile_time, speedup = measure_perf(
        input_size=input_size,
        padding=padding,
        warmup_iters=10,
        perf_iters=100,
        device=device,
        dtype=dtype)
    print(f'{input_size}, {padding}, {device}, {dtype}, {eager_time:.7f}, {compile_time:.7f}, {speedup:.3f}')

