import torch
import torch.utils.benchmark as benchmark

torch.set_num_threads(1)

def run_tests(test_cases):
    for idx, test_case in enumerate(test_cases):
        op, input_shape, kwargs, num_iters = test_case
        input = torch.randn(*input_shape, device='mps')

        def run():
            output = op(input, **kwargs)

        timer = benchmark.Timer(
            stmt='run()',
            globals=locals(),
            num_threads=1,
        )
        for _ in range(2):
            t_ms = timer.timeit(num_iters).mean * 1000
        print(f"{idx}: {t_ms:0.6f} ms, {op.__name__}, {input_shape}, {kwargs}")

test_cases = [
    # (op, input_shape, kwargs, num_iters)
    (torch.nn.functional.max_pool3d, (3, 2, 2, 2), dict(kernel_size=2), 1000),
    (torch.nn.functional.max_pool3d, (3, 10, 10, 10), dict(kernel_size=5), 1000),
    (torch.nn.functional.max_pool3d, (3, 100, 100, 100), dict(kernel_size=5), 100),
    (torch.nn.functional.max_pool3d, (3, 200, 200, 200), dict(kernel_size=5), 100),
    (torch.nn.functional.max_pool3d, (10, 10, 100, 100, 100), dict(kernel_size=4, padding=1), 10),
    (torch.nn.functional.max_pool3d, (10, 10, 100, 100, 100), dict(kernel_size=50, padding=20), 10),
    (torch.nn.functional.max_pool3d, (10, 10, 100, 100, 100), dict(kernel_size=50, padding=20, dilation=1), 10),
    (torch.nn.functional.max_pool3d, (10, 10, 100, 100, 100), dict(kernel_size=50, padding=20, dilation=1, stride=40), 10),
] + [
    (torch.nn.functional.max_pool3d, (10, 10, n, n, n), dict(kernel_size=2), 10) for n in range(10, 101, 20)
] + [
    (torch.nn.functional.max_pool3d, (10, 10, n, n, n), dict(kernel_size=2, dilation=2), 10) for n in [10, 50, 100]
]

print('===================')
print('max_pool3d')
print('===================')
run_tests(test_cases)

test_cases = [
    (torch.nn.functional.max_pool2d, (3, 2, 2), dict(kernel_size=2), 1000),
    (torch.nn.functional.max_pool2d, (3, 10, 10), dict(kernel_size=5), 1000),
    (torch.nn.functional.max_pool2d, (3, 100, 100), dict(kernel_size=5), 1000),
    (torch.nn.functional.max_pool2d, (3, 1000, 1000), dict(kernel_size=5), 100),
    (torch.nn.functional.max_pool2d, (3, 2000, 2000), dict(kernel_size=5), 100),
    (torch.nn.functional.max_pool2d, (10, 10, 1000, 1000), dict(kernel_size=4, padding=1), 10),
    (torch.nn.functional.max_pool2d, (10, 10, 1000, 1000), dict(kernel_size=100, padding=50), 10),
    (torch.nn.functional.max_pool2d, (10, 10, 1000, 1000), dict(kernel_size=250, padding=50, dilation=1), 10),
] + [
    (torch.nn.functional.max_pool2d, (10, 10, n, n), dict(kernel_size=2), 10) for n in range(100, 1001, 200)
] + [
    (torch.nn.functional.max_pool2d, (10, 10, n, n), dict(kernel_size=2, dilation=2), 10) for n in [100, 500, 1000]
]

print('===================')
print('max_pool2d')
print('===================')
run_tests(test_cases)
