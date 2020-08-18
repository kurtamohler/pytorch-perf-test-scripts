import torch
import time

torch.set_num_threads(1)
torch.manual_seed(0)

num_iters = 100

print('device, dtype, size, seconds_per_iter')
for device in ['cpu']:
    for dtype in [torch.float, torch.double]:
        for size in [(10, 10), (100, 100), (200, 200), (500, 500), (1000, 1000)]:
            a = torch.randn(*size, dtype=dtype, device=device)
            start_time = time.time()
            for i in range(num_iters):
                e, v = torch.eig(a, True)
            end_time = time.time()

            total_time = end_time - start_time
            seconds_per_iter = total_time / num_iters

            print(f'{device}, {dtype}, {size}, {seconds_per_iter}')
                
