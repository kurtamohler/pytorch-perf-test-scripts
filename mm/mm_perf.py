import torch
import time

def measure_mm_perf(dim_m, dim_n, dim_k, dtype, device, timed_iters):
    if dtype == torch.int32 or dtype == torch.int64:
        a = torch.randint(0xdeadbeef, [dim_m, dim_n], dtype=dtype, device=device)
        b = torch.randint(0xdeadbeef, [dim_n, dim_k], dtype=dtype, device=device)
    else:
        a = torch.rand([dim_m, dim_n], dtype=dtype, device=device)
        b = torch.rand([dim_n, dim_k], dtype=dtype, device=device)

    for warmup_iter in range(2):
        start_time = time.time()
        for timed_iter in range(timed_iters):
            ab = a.mm(b)
        total_time = time.time() - start_time
    return total_time/timed_iters


timed_iters = 100
print("dtype m n k runtime")
for dtype in [torch.float32, torch.float64]:
    for dim_m in [10, 100, 1000, 10_000]:
        for dim_n in [10, 100, 1000, 10_000]:
            for dim_k in [10, 100, 1000, 10_000]:
                runtime = measure_mm_perf(dim_m, dim_n, dim_k, dtype, 'cuda', timed_iters)
                print("%s %d %d %d %f" % (dtype, dim_m, dim_n, dim_k, runtime))

