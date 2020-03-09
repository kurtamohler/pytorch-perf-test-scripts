import time
import torch

torch.set_num_threads(1)

def block_diag_workaround(*arrs):
    shapes = torch.tensor([a.shape for a in arrs])
    out = torch.zeros(torch.sum(shapes, dim=0).tolist(), dtype=arrs[0].dtype, device=arrs[0].device)
    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out

def measure_block_diag_perf(num_mats, mat_dim_size, iters, dtype, device):
    if dtype in [torch.float32, torch.float64]:
        mats = [torch.rand(mat_dim_size, mat_dim_size, dtype=dtype, device=device) for i in range(num_mats)]
    else:
        mats = [torch.randint(0xdeadbeef, (mat_dim_size, mat_dim_size), dtype=dtype, device=device) for i in range(num_mats)]

    # do one warmup iteration
    for _ in range(2):
        torch_time_start = time.time()
        for i in range(iters):
            torch_result = torch.block_diag(*mats)
        torch_time = time.time() - torch_time_start

        workaround_time_start = time.time()
        for i in range(iters):
            workaround_result = block_diag_workaround(*mats)
        workaround_time = time.time() - workaround_time_start

    if not torch_result.equal(workaround_result):
        print("Results do not match!!")
        exit(1)
    return torch_time, workaround_time

iters = 10
print("data_type num_mats mat_dim_size torch_time workaround_time torch_speedup")
for device in ['cpu', 'cuda']:
    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
    # for dtype in [torch.float64]:
        for num_mats in [2, 8, 32]:
        # for num_mats in [8]:
            for mat_dim_size in [16, 64, 256, 512]:
            # for mat_dim_size in [1024]:
                torch_time, workaround_time = measure_block_diag_perf(num_mats, mat_dim_size, iters, dtype, device)
                torch_time /= iters
                workaround_time /= iters
                torch_speedup = workaround_time / torch_time
                print('%s %s %d %d %f %f %f' % (device, dtype, num_mats, mat_dim_size, torch_time, workaround_time, torch_speedup))


