import torch
import pandas as pd



dtype = torch.float
index_dtype = torch.long

accumulate = True
test_cases = []

test_cases += [
    # (self, index, value)
    (
        torch.zeros(4, 5, dtype=torch.float),
        (torch.tensor(0, dtype=index_dtype),),
        torch.tensor(1, dtype=dtype),
    ),
    (
        torch.zeros(4, 5, dtype=torch.float),
        (torch.tensor([0], dtype=index_dtype),),
        torch.tensor(1, dtype=dtype),
    ),
    (
        torch.zeros(4, 5, dtype=torch.float),
        (torch.tensor([[0]], dtype=index_dtype),),
        torch.tensor(1, dtype=dtype),
    ),
    (
        torch.zeros(4, 5, dtype=torch.float),
        (torch.tensor([0], dtype=index_dtype),),
        torch.tensor(1, dtype=dtype),
    ),

    # Try expanding the indices
    (
        torch.zeros(4, 5, dtype=torch.float),
        (
            torch.tensor([0], dtype=index_dtype),
            torch.tensor([0, 1, 2, 3, 4], dtype=index_dtype),
        ),
        torch.tensor([1], dtype=dtype),
    ),
    (
        torch.zeros(4, 5, dtype=torch.float),
        (
            torch.tensor([0], dtype=index_dtype),
            torch.tensor([0, 1, 2, 3, 4], dtype=index_dtype),
        ),
        torch.tensor([[1]], dtype=dtype),
    ),
    # Try expanding the value tensor
    (
        torch.zeros(4, 5, dtype=torch.float),
        (torch.tensor(0, dtype=index_dtype),),
        torch.tensor(1, dtype=dtype).unsqueeze(-1).expand((5,)),
    ),
    (
        torch.zeros(4, 5, dtype=torch.float),
        (torch.tensor([0], dtype=index_dtype),),
        torch.tensor(1, dtype=dtype).unsqueeze(-1).expand((5,)),
    ),
    (
        torch.zeros(4, 5, dtype=torch.float),
        (torch.tensor([[0]], dtype=index_dtype),),
        torch.tensor(1, dtype=dtype).unsqueeze(-1).expand((5,)),
    ),
]

test_cases += [

    # In this case, I think I could try adding another entry to `index`
    # so that it expands correctly
    (
        torch.zeros(4, 5, dtype=torch.float),
        (
            torch.tensor([0, 1], dtype=index_dtype),
        ),
        torch.tensor(1, dtype=dtype),
    ),
    (
        torch.zeros(4, 5, dtype=torch.float),
        (
            torch.tensor([0, 1], dtype=index_dtype),
        ),
        torch.tensor([1], dtype=dtype),
    ),
    (
        torch.zeros(4, 5, dtype=torch.float),
        (
            torch.tensor([0, 1], dtype=index_dtype),
        ),
        torch.tensor([1, 1, 1, 1, 1], dtype=dtype),
    ),

    # Like the following. Since this one works, I'm pretty sure this means
    # that the way `index_put` CUDA accumulate is adding in a null tensor
    # for missing dims is incorrect

    (
        torch.zeros(4, 5, dtype=torch.float),
        (
            torch.tensor([0, 1], dtype=index_dtype),
            torch.tensor([[0], [1], [2], [3], [4]], dtype=index_dtype),
        ),
        torch.tensor(1, dtype=dtype),
    ),

    (
        torch.zeros(4, 5, dtype=torch.float),
        (
            torch.tensor([[0], [1]], dtype=index_dtype),
            torch.tensor([0, 1, 2, 3, 4], dtype=index_dtype),
        ),
        torch.tensor(1, dtype=dtype),
    ),

    (
        torch.zeros(4, 5, dtype=torch.float),
        (
            torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=index_dtype),
            torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=index_dtype),
        ),
        torch.tensor(1, dtype=dtype),
    ),
    (
        torch.zeros(4, 5, dtype=torch.float),
        (
            #torch.tensor([[0], [1]], dtype=index_dtype),
            torch.tensor([[0], [1]], dtype=index_dtype),
            torch.tensor([0, 1, 2, 3, 4], dtype=index_dtype)
        ),
        torch.tensor(1, dtype=dtype),
    ),
]


# I think that if one of the elements of the tuple `index` has more than
# one dimension, then that element represents indices into the next two
# dimensions, rather than just one. Furthermore, we have to look at the
# sizes of the dimensions in backward order. For instance:
#
# (
#   [[0], [3], [2], [1]]

test_cases += [
    (
        torch.zeros(5, 5, dtype=torch.float),
        (torch.tensor([0, 3, 2, 1], dtype=index_dtype),),
        torch.arange(5).to(dtype) + 1,
    ),
    (
        torch.zeros(5, 5, dtype=torch.float),
        (torch.tensor([0, 3, 2, 1], dtype=index_dtype).unsqueeze(-1),),
        torch.arange(5).to(dtype) + 1,
    ),
]

test_cases += [
    (
        torch.zeros(5, 5, dtype=torch.float),
        (torch.tensor(0, dtype=index_dtype),),
        torch.arange(5).to(dtype) + 1,
    ),
    (
        torch.zeros(5, 5, dtype=torch.float),
        (torch.tensor([0], dtype=index_dtype),),
        torch.arange(5).to(dtype) + 1,
    ),
    (
        torch.zeros(5, 5, dtype=torch.float),
        (torch.tensor([[0]], dtype=index_dtype),),
        torch.arange(5).to(dtype) + 1,
    ),
    (
        torch.zeros(5, 5, 5, dtype=torch.float),
        (torch.tensor(0, dtype=index_dtype),),
        torch.arange(5).to(dtype) + 1,
    ),
    (
        torch.zeros(5, 5, 5, dtype=torch.float),
        (torch.tensor([0], dtype=index_dtype),),
        torch.arange(5).to(dtype) + 1,
    ),
    (
        torch.zeros(5, 5, 5, dtype=torch.float),
        (torch.tensor([[0]], dtype=index_dtype),),
        torch.arange(5).to(dtype) + 1,
    ),
    (
        torch.zeros(5, 5, 5, dtype=torch.float),
        (torch.tensor(0, dtype=index_dtype),),
        torch.arange(25).reshape((5,5)).to(dtype) + 1,
    ),
    (
        torch.zeros(5, 5, 5, dtype=torch.float),
        (torch.tensor([0], dtype=index_dtype),),
        torch.arange(25).reshape((5,5)).to(dtype) + 1,
    ),
    (
        torch.zeros(5, 5, 5, dtype=torch.float),
        (torch.tensor([[0]], dtype=index_dtype),),
        torch.arange(25).reshape((5,5)).to(dtype) + 1,
    ),
    (
        torch.zeros(5, 5, dtype=torch.float),
        (torch.tensor([0, 3, 2, 1], dtype=index_dtype),),
        torch.arange(5).to(dtype) + 1,
    ),
    (
        torch.zeros(5, 5, dtype=torch.float),
        (torch.tensor([[0], [3], [2], [1]], dtype=index_dtype),),
        torch.arange(5).to(dtype) + 1,
    ),
    (
        torch.zeros(5, 5, dtype=torch.float),
        (torch.tensor([[0], [3], [2], [1]], dtype=index_dtype),),
        torch.tensor(1, dtype=dtype),
    ),
    (
        torch.zeros(4, 5, dtype=torch.float),
        (torch.tensor([0, 3, 2, 1], dtype=index_dtype),),
        torch.arange(5).to(dtype) + 1,
    ),
    (
        torch.zeros(4, 5, dtype=torch.float),
        (torch.tensor([[0], [3], [2], [1]], dtype=index_dtype),),
        torch.arange(5).to(dtype) + 1,
    ),
    (
        torch.zeros(4, 5, dtype=torch.float),
        (torch.tensor([[0], [3], [2], [1]], dtype=index_dtype),),
        torch.tensor(1, dtype=dtype),
    ),
    (
        torch.zeros(5, 5, 5, dtype=torch.float),
        (torch.tensor([0, 3, 2, 1], dtype=index_dtype),),
        torch.arange(5).to(dtype) + 1,
    ),
    (
        torch.zeros(5, 5, 5, dtype=torch.float),
        (torch.tensor([[0], [3], [2], [1]], dtype=index_dtype),),
        torch.arange(5).to(dtype) + 1,
    ),
    (
        torch.zeros(5, 5, 5, dtype=torch.float),
        (torch.tensor([[0], [3], [2], [1]], dtype=index_dtype),),
        torch.arange(5).to(dtype) + 1,
    ),
    (
        torch.zeros(5, 5, dtype=torch.float),
        (
            torch.tensor([0, 0, 2, 3], dtype=index_dtype),
            torch.tensor([1, 1, 2, 4], dtype=index_dtype),
        ),
        torch.tensor(1, dtype=dtype),
    ),
    (
        torch.zeros(4, 1, 1, dtype=torch.float),
        (
            torch.tensor([0, 1, 2, 3], dtype=index_dtype),
            torch.tensor([0], dtype=index_dtype),
            torch.tensor([0], dtype=index_dtype),
        ),
        torch.tensor(1, dtype=dtype),
    ),
    (
        torch.zeros(4, 1, 1, dtype=torch.float),
        (
            torch.tensor([0, 1, 2, 3], dtype=index_dtype),
            torch.tensor([0], dtype=index_dtype),
        ),
        torch.arange(4).to(dtype) + 1,
    ),
    (
        torch.zeros(4, 1, 1, dtype=torch.float),
        (
            torch.tensor([0, 1, 2, 3], dtype=index_dtype),
            torch.tensor([0], dtype=index_dtype),
        ),
        torch.arange(1).to(dtype) + 1,
    ),
]

df = pd.DataFrame(
    columns=['self_size', 'index_size', 'value_size', 'cpu_works', 'cuda_works'])

for test_case_idx, (self, index, value) in enumerate(test_cases):
    print('=============================')
    print(f'test case {test_case_idx}')
    print('=============================')
    print()
    cpu_works = True
    cuda_works = 'y'

    self = self.to('cpu')
    index = tuple(i.to('cpu') for i in index)
    value = value.to('cpu')

    try:
        res_cpu = torch.index_put(self, index, value, accumulate=accumulate)

    except Exception as e:
        cpu_works = False
        res_cpu = None
        print(f'CPU error: {e}')

    self = self.to('cuda')
    index = tuple(i.to('cuda') for i in index)
    value = value.to('cuda')

    try:
        res_cuda = torch.index_put(self, index, value, accumulate=accumulate)
        if cpu_works:
            if (res_cpu.size() != res_cuda.size()) or (res_cpu.ne(res_cuda.to('cpu')).any()):
                    cuda_works = 'unequal'

    except Exception as e:
        res_cuda = None
        cuda_works = 'error'
        print(f'CUDA error: {e}')


    self_size = tuple(self.size())
    index_size = tuple(tuple(i.size()) for i in index)
    value_size = tuple(value.size())
    cpu_works = 'y' if cpu_works else 'error'

    df.loc[test_case_idx] = [self_size, index_size, value_size, cpu_works, cuda_works]
    print()
    print('CPU result:')
    print(res_cpu)
    print()
    print('CUDA result:')
    print(res_cuda)
    print()

print('=============================')
print('summary')
print('=============================')
print()

print(df)

