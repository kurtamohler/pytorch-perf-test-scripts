import torch
import time

torch.set_num_threads(1)

timed_iters = 100
max_mask_int = 10_000_000

print('device dtype tensor_size mask_true_ratio masked_select_time index_time')

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
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    total_time = time.time() - start_time

                    # Compare with index performance, since masked_select will do the
                    # exact same thing if no broadcasting is needed
                    start_index_time = time.time()
                    for i in range(timed_iters):
                        a_indexed = a[mask]
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    total_index_time = time.time() - start_index_time


                time_per_iter = total_time / timed_iters
                time_per_index_iter = total_index_time / timed_iters

                if device == 'cpu':
                    a_masked_test = torch.masked_select(a.cuda(), mask.cuda())
                else:
                    a_masked_test = torch.masked_select(a.cpu(), mask.cpu())

                matching = torch.all(a_masked_test.cpu().eq(a_masked.cpu()))
                if not matching:
                    print('CPU and CUDA do not match')
                    exit(1)

                matching = torch.all(a_indexed.eq(a_masked))
                if not matching:
                    print('masked_select and index do not match')
                    exit(1)

                print("%s %s %s %f %f %f" % (
                    device,
                    dtype,
                    tensor_size,
                    mask_true_ratio,
                    time_per_iter,
                    time_per_index_iter
                ))
