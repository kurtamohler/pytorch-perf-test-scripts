import torch
import random

def gen_with_randn(mat_size, dtype):
    return torch.randn(mat_size, mat_size, dtype=dtype)

def gen_with_svd(mat_size, dtype):
    x = torch.randint(-5, 5, (mat_size, mat_size)).to(torch.cdouble)
    if dtype.is_complex:
        x += 1j * torch.randint(-5, 5, (mat_size, mat_size)).to(torch.cdouble)

    # Need at least one linearly dependent pair of rows
    x[1].copy_(x[0])

    k = torch.matrix_rank(x)
    assert k < mat_size

    u, _, v = x.svd()

    s = torch.randint(1, 10, (mat_size,)).to(torch.cdouble)
    if dtype.is_complex:
        s += 1j * torch.randint(1, 10, (mat_size,)).to(torch.cdouble)
    s[-k:] = 0

    matrix = (u * s.unsqueeze(-2)) @ v.transpose(-1, -2).conj()

    return matrix

def gen_with_matmul_floats(mat_size, dtype):
    a = torch.randn(mat_size, 1, dtype=dtype)
    b = torch.randn(1, mat_size, dtype=dtype)
    return a @ b

def gen_with_matmul_ints_base(mat_size, dtype, avoid_svd_eq_0, prenormalize=False):
    lo = -5
    hi = 5
    a_size = (mat_size, 1)
    b_size = (1, mat_size)
    if dtype.is_complex:
        a = torch.randint(lo, hi, a_size).to(dtype).add(
            1j * torch.randint(lo, hi, a_size).to(dtype))
        b = torch.randint(lo, hi, b_size).to(dtype).add(
            1j * torch.randint(lo, hi, b_size).to(dtype))

        if avoid_svd_eq_0:
            a[a.eq(0)] = random.choice([hi, 1j * hi])
            b[b.eq(0)] = random.choice([hi, 1j * hi])
    else:
        assert False

    if prenormalize:
        a /= a.norm()
        b /= b.norm()
    return a @ b

def gen_with_matmul_ints(mat_size, dtype):
    return gen_with_matmul_ints_base(mat_size, dtype, False)


def gen_with_matmul_ints_avoid_svd_eq_0(mat_size, dtype, prenormalize=False):
    return gen_with_matmul_ints_base(mat_size, dtype, True, prenormalize)

def gen_with_matmul_ints_avoid_svd_eq_0_rank_2(mat_size, dtype):
    return gen_with_matmul_ints_avoid_svd_eq_0(mat_size, dtype) + gen_with_matmul_ints_avoid_svd_eq_0(mat_size, dtype)

def gen_prenorm_sum(mat_size, dtype, rank):
    res = torch.zeros(mat_size, mat_size, dtype=dtype)

    for r in range(rank):
        u = torch.torch.rand(mat_size, dtype=dtype)
        u /= u.norm()

        v = torch.torch.rand(mat_size, dtype=dtype)
        v /= v.norm()

        res += u.unsqueeze(-1) * v.unsqueeze(-2)

    return res

def gen_prenorm_sum_rank_1(mat_size, dtype):
    return gen_prenorm_sum(mat_size, dtype, 1)

def gen_prenorm_sum_rank_2(mat_size, dtype):
    return gen_prenorm_sum(mat_size, dtype, 2)

def gen_with_matmul_ints_prenorm_sum(mat_size, dtype, rank):
    res = torch.zeros(mat_size, mat_size, dtype=dtype)

    for r in range(rank):
        res += gen_with_matmul_ints_avoid_svd_eq_0(mat_size, dtype, prenormalize=True)

    return res

def gen_with_matmul_ints_prenorm_sum_rank_1(mat_size, dtype):
    return gen_with_matmul_ints_prenorm_sum(mat_size, dtype, 1)

def gen_with_matmul_ints_prenorm_sum_rank_2(mat_size, dtype):
    return gen_with_matmul_ints_prenorm_sum(mat_size, dtype, 2)

gen_funcs = [
    gen_with_randn,
    gen_with_svd,
    gen_with_matmul_floats,
    gen_with_matmul_ints,
    gen_with_matmul_ints_avoid_svd_eq_0,
    gen_with_matmul_ints_avoid_svd_eq_0_rank_2,
    gen_prenorm_sum_rank_1,
    gen_prenorm_sum_rank_2,
    gen_with_matmul_ints_prenorm_sum_rank_1,
    gen_with_matmul_ints_prenorm_sum_rank_2,
]

num_tries = 10000
mat_size = 4

print('| func_name | ratio det is exactly 0 | ratio singular value 0 | ratio acceptable |')
print('| --- | --- | --- | --- |')
for gen_func in gen_funcs:
    torch.manual_seed(0)
    random.seed(0)
    num_det_eq_0 = 0
    num_has_svd_0 = 0
    num_acceptable = 0

    for _ in range(num_tries):
        m = gen_func(mat_size, torch.cdouble)

        if m.det().eq(0).item():
            num_det_eq_0 += 1

        if m.svd()[1].eq(0).any():
            num_has_svd_0 += 1

        if m.det().eq(0).item() and m.svd()[1].ne(0).all().item():
            num_acceptable += 1

    print(f'| {gen_func.__name__} | {num_det_eq_0 / num_tries} | {num_has_svd_0 / num_tries} | {num_acceptable / num_tries} |')


