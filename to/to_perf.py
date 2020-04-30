import torch
import time

torch.set_num_threads(1)

timed_iters = 10

print("device in_dtype out_dtype tensor_size time_per_iter")
for device in ['cpu']:
    for dtype in [torch.long, torch.int, torch.bool, torch.uint8]:
        for out_dtype in [torch.long, torch.int, torch.bool, torch.uint8]:
            for tensor_size in [1_000_000]:
                if dtype == torch.float:
                    a = torch.rand(tensor_size, dtype=dtype, device=device)
                elif dtype in [torch.int, torch.long]:
                    a = torch.randint(100, [tensor_size], dtype=dtype, device=device)
                else:
                    a = torch.randint(2, [tensor_size], dtype=dtype, device=device)
                for warmup_iter in range(2):
                    start_time = time.time()
                    for i in range(timed_iters):
                        a_cast = a.to(out_dtype)
                    total_time = time.time() - start_time

                time_per_iter = total_time / timed_iters

                print("%s %s %s %s %f" % (
                    device,
                    dtype,
                    out_dtype,
                    tensor_size,
                    time_per_iter
                ))
