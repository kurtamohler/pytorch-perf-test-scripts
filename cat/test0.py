import torch

def run(device):
    dtype = torch.int
    return torch.cat([
        torch.arange(100_000_000, device=device, dtype=dtype).reshape((100, 1000, 1000)),
        torch.tensor([], device=device, dtype=dtype),
        torch.arange(50_000_000, device=device, dtype=dtype).reshape((100, 500, 1000)),
    ], dim=1)


r_cpu = run('cpu')
r_mps = run('mps')

assert (r_cpu == r_mps.cpu()).all()

print(r_cpu.shape)
