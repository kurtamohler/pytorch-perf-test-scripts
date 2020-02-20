import torch
import time

torch.set_num_threads(1)

timed_iters = 100
max_mask_int = 10_000_000

print('device dtype tensor_size mask_true_ratio time_per_iter')

for device in ['cpu', 'cuda']:
    for dtype in [torch.float64]:
        for tensor_size in [1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
            for mask_true_ratio in [0., 0.25, 0.5, 0.75, 1.]:
                a = torch.rand(tensor_size, dtype=dtype, device=device)

                # Create a mask with approximately the correct ratio
                # of randomly placed True/False values
                mask_ints = torch.randint(max_mask_int, [tensor_size], dtype=dtype, device=device)
                cutoff_int = int(max_mask_int * mask_true_ratio)
                mask = mask_ints < cutoff_int


                for warmup_iter in range(2):
                    start_time = time.time()
                    for i in range(timed_iters):
                        a_masked = torch.masked_select(a, mask)
                    total_time = time.time() - start_time

                time_per_iter = total_time / timed_iters

                print("%s %s %d %f %f" % (
                    device,
                    dtype,
                    tensor_size,
                    mask_true_ratio,
                    time_per_iter
                ))