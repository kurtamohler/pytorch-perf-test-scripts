import torch
import time

torch.set_num_threads(1)

timed_iters = 100

for device in ['cpu']:
    for dtype in [torch.float, torch.int, torch.bool, torch.uint8]:
        out_dtype = torch.int if dtype != torch.float else torch.float
        for tensor_size in [1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
            if dtype == torch.float:
                a = torch.rand(tensor_size, dtype=dtype, device=device)
            elif dtype == torch.int:
                a = torch.randint(100, [tensor_size], dtype=dtype, device=device)
            else:
                a = torch.randint(2, [tensor_size], dtype=dtype, device=device)
            for warmup_iter in range(2):
                start_time = time.time()
                for i in range(timed_iters):
                    a_sum = a.sum(dtype=out_dtype)
                # if device == 'cuda':
                #     torch.cuda.synchronize()
                total_time = time.time() - start_time

            time_per_iter = total_time / timed_iters

            print("%s %s %s %f" % (
                device,
                dtype,
                tensor_size,
                time_per_iter
            ))
