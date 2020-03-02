import torch
import time

torch.set_num_threads(1)

timed_iters = 10

print('device dtype num_elements seconds')
for device in ['cpu', 'cuda']:
    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
        if device == 'cpu':
            tensor_sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
        else:
            tensor_sizes = [1_000_000, 10_000_000, 100_000_000]

        for num_elements in tensor_sizes:
            if dtype in [torch.float32, torch.float64]:
                a = torch.rand(num_elements, dtype=dtype, device=device)
                b = torch.rand(1, dtype=dtype, device=device)[0]
            else:
                a = torch.randint(0xdeadbeef, [num_elements], dtype=dtype, device=device)
                b = torch.randint(0xdeadbeef, [1], dtype=dtype, device=device)[0]

            for warmup_iter in range(2):
                start_time = time.time()
                for i in range(timed_iters):
                    a_remainder = a.remainder(b)
                total_time = time.time() - start_time

            time_per_iter = total_time / timed_iters

            print('%s %s %d %f' % (
                device,
                dtype,
                num_elements,
                time_per_iter
            ))

