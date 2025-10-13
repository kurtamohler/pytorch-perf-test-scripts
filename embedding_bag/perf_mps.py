import torch
import torch.utils.benchmark as benchmark
import itertools

torch.set_num_threads(1)

def arg_list_to_string(args):
    return f"{[list(a.shape) if torch.is_tensor(a) else a for a in args]}"

def convert_args_kwargs_to_device(args, kwargs, device):
    args_out = [a.to(device) if torch.is_tensor(a) else a for a in args]
    kwargs_out = {key: val.to(device) if torch.is_tensor(val) else val for key, val in kwargs.items()}
    return args_out, kwargs_out


def run_tests(test_cases):
    print('idx: cpu time, mps time, speedup, op, args, kwargs')
    print('-----------------------------------------')
    for idx, test_case in enumerate(test_cases):
        op, args, kwargs, num_iters = test_case

        args_cpu, kwargs_cpu = convert_args_kwargs_to_device(args, kwargs, 'cpu')
        args_mps, kwargs_mps = convert_args_kwargs_to_device(args, kwargs, 'mps')

        def run_cpu():
            output = op(*args_cpu, **kwargs_cpu)

        def run_mps():
            output = op(*args_mps, **kwargs_mps)

        timer_cpu = benchmark.Timer(
            stmt='run_cpu()',
            globals=locals(),
            num_threads=1,
        )
        timer_mps = benchmark.Timer(
            stmt='run_mps()',
            globals=locals(),
            num_threads=1,
        )
        for _ in range(2):
            t_cpu_ms = timer_cpu.timeit(num_iters).median * 1000
        for _ in range(2):
            t_mps_ms = timer_mps.timeit(num_iters).median * 1000

        speedup = t_cpu_ms / t_mps_ms
        print(f"{idx}: {t_cpu_ms:0.6f} ms, {t_mps_ms:0.6f} ms, {speedup:0.2f}, {op.__name__}, {arg_list_to_string(args)}, {kwargs}")

# `sizes` is a list of input sizes you want to test. Each key in `kwargs` is one
# of the kwargs of the given op, and its value is a list of the different values
# that will be given for that kwarg.
def params_product(op, num_iters, sizes, **kwargs):
    return [
        (op, values[0], dict(zip(kwargs.keys(), values[1:])), num_iters) for values in itertools.product(
            sizes, *kwargs.values()
        )
    ]

op = torch.nn.functional.embedding_bag

test_cases = [
    [op, [
        torch.randint(0, 20, (10, 4)),
        torch.randn(20, 5),
        #torch.arange(0, 40, 10),
    ], {}, 1000 ],
    [op, [
        torch.randint(0, 20, (40,)),
        torch.randn(20, 5),
        torch.arange(0, 40, 10),
    ], {}, 1000 ],
    [op, [
        torch.randint(0, 20000, (40,)),
        torch.randn(20000, 5),
        torch.arange(0, 40, 10),
    ], {}, 1000 ],
    [op, [
        torch.randint(0, 20, (40000,)),
        torch.randn(20, 5),
        torch.arange(0, 40000, 10),
    ], {}, 1000 ],
    [op, [
        torch.randint(0, 20000, (40000,)),
        torch.randn(20000, 5),
        torch.arange(0, 40000, 10),
    ], {}, 1000 ],
]

run_tests(test_cases)
