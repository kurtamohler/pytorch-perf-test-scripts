import torch
import torch.utils.benchmark as benchmark
import itertools

torch.set_num_threads(1)

def run_tests(test_cases):
    for idx, test_case in enumerate(test_cases):
        op, input_shape, kwargs, num_iters = test_case
        input = torch.randn(*input_shape, device='mps')

        # First test if MPS result matches CPU. If not, mark this case as a
        # mismatch and don't measure performance.
        output_mps = op(input, **kwargs)
        output_cpu = op(input.cpu(), **kwargs)

        if not torch.allclose(output_mps.cpu(), output_cpu, atol=1e-2, rtol=1e-2):
            print(f"{idx}: __mismatch__, {op.__name__}, {input_shape}, {kwargs}")
            continue

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


# `sizes` is a list of input sizes you want to test. Each key in `kwargs` is one
# of the kwargs of the given op, and its value is a list of the different values
# that will be given for that kwarg.
def params_product(op, num_iters, sizes, **kwargs):
    return [
        (op, values[0], dict(zip(kwargs.keys(), values[1:])), num_iters) for values in itertools.product(
            sizes, *kwargs.values()
        )
    ]

op = torch.nn.functional.avg_pool2d

test_cases = (
    params_product(
        op, 100,
        [
            (4, 2, 3),
            (5, 2, 3),
            (50, 2, 3),
            (4, 1, 2, 3),
            (4, 4, 2, 3),
            (2, 2, 4, 6),
            (5, 40, 60),
            (2, 2, 40, 60),
            (3, 2, 3),
            (2, 2, 3),
            (1, 2, 3),
            (3, 3, 2, 3),
            (3, 40, 60),
            (4, 40, 60),
            (1, 1, 40, 60),
        ],
        stride=[[2, 3]],
        kernel_size=[[1, 3]],
        ceil_mode=[False, True],
        divisor_override=[None, 7],
    )
    + params_product(
        op, 1000, [(3, 2, 2)], kernel_size=[2],
        ceil_mode=[False, True],
        divisor_override=[None, 7],
    )
    + params_product(
        op, 1000, [(3, 10, 10)], kernel_size=[5],
        ceil_mode=[False, True],
        divisor_override=[None, 7],
    )
    + params_product(
        op, 1000, [(3, 100, 100)], kernel_size=[5],
        ceil_mode=[False, True],
        divisor_override=[None, 7],
    )
    + params_product(
        op, 100, [(3, 1000, 1000), (3, 2000, 2000)], kernel_size=[5],
        stride=[None, 1, 2, 4],
        padding=[0, 1, 2],
        ceil_mode=[False, True],
        divisor_override=[None, 7],
    )
    + params_product(
        op, 100, [(10, 10, 1000, 1000)], kernel_size=[4], padding=[1],
        stride=[None, 1],
        ceil_mode=[False, True],
        divisor_override=[None, 7],
    )
    + params_product(
        op, 10, [(10, 10, 1000, 1000)], kernel_size=[100], padding=[50],
        ceil_mode=[False, True],
        divisor_override=[None, 7],
    )
    + params_product(
        op, 20, [(10, 10, 1000, 1000)], kernel_size=[250], padding=[50],
        ceil_mode=[False, True],
        divisor_override=[None, 7],
    )
    + params_product(
        op, 100, [(10, 10, n, n) for n in [100, 300, 500, 700, 900]], kernel_size=[2],
        ceil_mode=[False, True],
        divisor_override=[None, 7],
    )
    + params_product(
        op, 100, [(10, 10, n, n) for n in [100, 500, 1000]], kernel_size=[2],
        ceil_mode=[False, True],
        divisor_override=[None, 7],
    )
    + params_product(
        op, 100, [(10, 1000, 1000)],
        kernel_size=[2, 4, 8, 16],
        padding=[1],
        stride=[1],
        ceil_mode=[False, True],
        divisor_override=[None, 7],
    )
    + params_product(
        op, 100, [(10, 1000, 1000)],
        kernel_size=[4],
        padding=[0, 1],
        stride=[(1, 1), (1, 4), (4, 1)],
        ceil_mode=[False, True],
        divisor_override=[None, 7],
    )
    + params_product(
        op, 100, [(10, 1000, 1000)],
        kernel_size=[1],
        ceil_mode=[False, True],
        divisor_override=[None, 7],
    )
)

run_tests(test_cases)
