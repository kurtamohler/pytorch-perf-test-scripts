import torch
import torch.utils.benchmark as benchmark
import itertools

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
            t_ms = timer.timeit(num_iters).median * 1000
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


# `sizes` is a list of input sizes you want to test. Each key in `kwargs` is one
# of the kwargs of the given op, and its value is a list of the different values
# that will be given for that kwarg.
def params_product(op, num_iters, sizes, **kwargs):
    return [
        (op, values[0], dict(zip(kwargs.keys(), values[1:])), num_iters) for values in itertools.product(
            sizes, *kwargs.values()
        )
    ]

op = torch.nn.functional.max_pool2d

test_cases = (
    params_product(
        op, 1000, [(3, 2, 2)], kernel_size=[2],
        return_indices=[True, False],
    )
    + params_product(
        op, 1000, [(3, 10, 10)], kernel_size=[5],
        return_indices=[True, False],
    )
    + params_product(
        op, 1000, [(3, 100, 100)], kernel_size=[5],
        return_indices=[True, False],
    )
    + params_product(
        op, 100, [(3, 1000, 1000), (3, 2000, 2000)], kernel_size=[5],
        dilation=[1, 2, 4],
        stride=[None, 1, 2, 4],
        padding=[0, 1, 2],
        return_indices=[True, False],
    )
    + params_product(
        op, 100, [(10, 10, 1000, 1000)], kernel_size=[4], padding=[1],
        stride=[None, 1],
        return_indices=[True, False],
    )
    + params_product(
        op, 10, [(10, 10, 1000, 1000)], kernel_size=[100], padding=[50],
        return_indices=[True, False],
    )
    + params_product(
        op, 20, [(10, 10, 1000, 1000)], kernel_size=[250], padding=[50],
        return_indices=[True, False],
    )
    + params_product(
        op, 100, [(10, 10, n, n) for n in [100, 300, 500, 700, 900]], kernel_size=[2],
        return_indices=[True, False],
    )
    + params_product(
        op, 100, [(10, 10, n, n) for n in [100, 500, 1000]], kernel_size=[2],
        return_indices=[True, False],
    )
    + params_product(
        op, 100, [(10, 1000, 1000)],
        kernel_size=[2, 4, 8, 16],
        padding=[1],
        stride=[1],
        return_indices=[False, True],
    )
    + params_product(
        op, 100, [(10, 1000, 1000)],
        kernel_size=[4],
        padding=[0, 1],
        stride=[(1, 1), (1, 4), (4, 1)],
        return_indices=[False, True],
    )
    + params_product(
        op, 100, [(10, 1000, 1000)],
        kernel_size=[1],
        return_indices=[False, True],
    )
)


print('===================')
print('max_pool2d')
print('===================')
run_tests(test_cases)
