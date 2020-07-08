import torch
import numpy
import math
import pandas

torch.manual_seed(0)

def compare_torch_and_numpy(a, p_list):
    a_n = a.numpy()
    torch_results = []
    numpy_results = []
    matches = []
    for p in p_list:
        torch_result = a.norm(p=p)
        numpy_result = numpy.linalg.norm(a_n, ord=p)

        torch_results.append(torch_result.item())
        numpy_results.append(numpy_result)

        # Check if either absolute or relative diff is below
        # a threshold
        eps = 1e-5
        match = (torch_result - numpy_result).abs().lt(eps).item()
        if not match:
            match = ((torch_result - numpy_result) / numpy_result).abs().lt(eps).item()
        matches.append(match)
    
    df = pandas.DataFrame({
        'p': p_list,
        'torch_result': torch_results,
        'numpy_result': numpy_results,
        'equal': matches
    })
    return df.where(df.notnull(), None)

a = torch.randn(100) + 1j * torch.randn(100)
p_list = [None, math.inf, 2, 1, 0.5, 0, -0.5, -1, -2, -math.inf]
print('Complex vector norms:')
print(compare_torch_and_numpy(a, p_list))
print()

a = torch.randn(5, 20) + 1j * torch.randn(5, 20)
a = torch.tensor([[1+2j, 3j], [4, 3+5j]])
p_list = ['fro', 'nuc', math.inf, 2, 1, -1, -2, -math.inf]
print('Complex matrix norms:')
print(compare_torch_and_numpy(a, p_list))
