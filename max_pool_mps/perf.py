import torch
import timeit
import torch.utils.benchmark as benchmark

torch.set_num_threads(1)

test_cases = [
    # (op, input_shape, kwargs, num_iters)
    (torch.nn.functional.max_pool3d, (1, 1, 10, 10), dict(kernel_size=(1, 2, 2)), 100),
    (torch.nn.functional.max_pool3d, (1, 1, 100, 100), dict(kernel_size=(1, 2, 2)), 100),
    (torch.nn.functional.max_pool3d, (1, 1, 1000, 1000), dict(kernel_size=(1, 2, 2)), 100),
    (torch.nn.functional.max_pool2d, (1, 1, 2, 2), dict(kernel_size=2), 100),
    (torch.nn.functional.max_pool2d, (1, 1, 10, 10), dict(kernel_size=2), 100),
    (torch.nn.functional.max_pool2d, (1, 1, 100, 100), dict(kernel_size=2), 100),
    (torch.nn.functional.max_pool2d, (1, 1, 1000, 1000), dict(kernel_size=2), 100),
]

for test_case in test_cases:
    op, input_shape, kwargs, num_iters = test_case
    input = torch.randn(*input_shape, device='mps')

    def run():
        output = op(input, **kwargs)

    timer = benchmark.Timer(
        stmt='run()',
        globals=globals(),
        num_threads=1,
    )
    for _ in range(2):
        t_ms = timer.timeit(100).mean * 1000
    print(f"{t_ms:0.6f}: {op.__name__}, {input_shape}, {kwargs}")

