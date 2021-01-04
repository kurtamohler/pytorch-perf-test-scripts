# Since torch.linalg.norm's 'flatten' arg is meant to replace the N-D input
# support that torch.norm offers, it is important to check that it will not
# degrade performance when we upgrade torch.norm calls to torch.linalg.norm

import torch
import timeit

torch.set_num_threads(1)

numel_and_timeitnum_list = [
    (2 ** 8, 10000),
    (2 ** (8 * 2), 1000),
    (2 ** (8 * 3), 10),
]

for device in ['cpu', 'cuda']:
    for dtype in [torch.float32, torch.float64, torch.cfloat, torch.cdouble]:
        print("===============================")
        print(f"{device} {dtype}")
        print("===============================")
        print()
        print('ndim numel ord norm_time linalg_norm_time linalg_speedup')
        print()
        for ndim in [1, 2, 4, 8]:
            for numel, timeitnum in numel_and_timeitnum_list:
                dim_size = int(round(numel ** (1 / ndim)))
                input_size = [dim_size] * ndim
                for ord in [-float('inf'), -3.5, -2, -1, -0.5, 0, 0.5, 1, 2, 3.5, float('inf')]:
                    variables = {
                        'ord': ord,
                        'input_size': input_size,
                        'dtype': dtype,
                        'device': device,
                        'torch': torch,
                    }

                    torch.manual_seed(0)
                    linalg_norm_time = timeit.repeat(
                        stmt='torch.linalg.norm(input, ord, flatten=True)',
                        setup='input = torch.randn(input_size, dtype=dtype, device=device)',
                        repeat=2,
                        number=timeitnum,
                        globals=variables)[1]

                    torch.manual_seed(0)
                    norm_time = timeit.repeat(
                        stmt='torch.norm(input, p=ord)',
                        setup='input = torch.randn(input_size, dtype=dtype, device=device)',
                        repeat=2,
                        number=timeitnum,
                        globals=variables)[1]

                    linalg_speedup = norm_time / linalg_norm_time

                    print(f'{ndim} {numel} {ord} {norm_time} {linalg_norm_time} {linalg_speedup}')
            print()
        print()

