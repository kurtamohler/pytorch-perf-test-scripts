import torch
from torch.testing import make_tensor
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.opinfo.core import SampleInput
import torch.utils.benchmark as benchmark

from contextlib import contextmanager
from functools import partial

torch.set_num_threads(1)

@contextmanager
def temporary_repr(cls, repr):
    repr_restore = cls.__repr__
    cls.__repr__ = repr
    try:
        yield cls
    finally:
        cls.__repr__ = repr_restore

def maybe_truncate_str(s, n):
    if len(s) > n:
        return '"' + s[:n] + '..."'
    return s

def run_tests(op, dtype, num_iters):
    print('idx: cpu time, mps time, speedup, op, args, kwargs')
    print('-----------------------------------------')

    def get_samples(device):
        return op.sample_inputs(
            device,
            dtype,
            include_conjugated_inputs=dtype.is_complex and op.test_conjugated_samples,
            set_seed=True,
        )

    samples_cpu = get_samples('cpu')
    samples_mps = get_samples('mps')

    for idx, (sample_cpu, sample_mps) in enumerate(zip(samples_cpu, samples_mps)):
        args_cpu = [sample_cpu.input] + list(sample_cpu.args)
        kwargs_cpu = sample_cpu.kwargs

        args_mps = [sample_mps.input] + list(sample_mps.args)
        kwargs_mps = sample_mps.kwargs

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
            t_mps_ms = timer_mps.timeit(num_iters).median * 1000

        speedup = t_cpu_ms / t_mps_ms

        with temporary_repr(torch.Tensor, lambda self: f'tensor(shape{list(self.shape)})'):
            print(f"{idx}: {t_cpu_ms:0.6f} ms, {t_mps_ms:0.6f} ms, {speedup:0.2f}, {op.name}, {maybe_truncate_str(str(args_mps), 100)}, {kwargs_mps}")


# Get the op from the opinfos database and add some extra sample inputs for
# better performance coverage.
def get_op():
    op = next((op for op in op_db if op.name == "linalg.householder_product"), None)

    sample_inputs_orig = op.sample_inputs_func

    def sample_inputs(op_info, device, dtype, requires_grad, **kwargs):
        for sample in sample_inputs_orig(op_info, device, dtype, requires_grad, **kwargs):
            yield sample

        make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

        cases = [
            # [A_shape, tau_shape]

            [(10, 10), (10)],
            [(10, 10, 10), (10, 10)],
            [(1000, 10, 10), (1000, 10)],
            [(500, 500), (500)],
        ]

        for A_shape, tau_shape in cases:
            A = make_arg(A_shape)
            tau = make_arg(tau_shape)
            yield SampleInput(A, tau)

    op.sample_inputs_func = sample_inputs

    return op

op = get_op()

run_tests(op, torch.float, 20)
