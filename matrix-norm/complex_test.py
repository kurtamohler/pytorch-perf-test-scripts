import torch
import numpy
import math
import pandas


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
        match = (torch_result - numpy_result).abs().lt(1e-5).all()
        matches.append(match.item())
    
    df = pandas.DataFrame({
        'p': p_list,
        'torch_result': torch_results,
        'numpy_result': numpy_results,
        'equal': matches
    })
    return df.where(df.notnull(), None)

a = torch.tensor([1+2j, 3j, 4, 3+5j])
p_list = [None, math.inf, 2, 1, 0.5, 0, -0.5, -1, -2, -math.inf]
print('Complex vector norms:')
print(compare_torch_and_numpy(a, p_list))
print()

a = torch.tensor([[1+2j, 3j], [4, 3+5j]])
p_list = ['fro', 'nuc', math.inf, 2, 1, -1, -2, -math.inf]
print('Complex matrix norms:')
print(compare_torch_and_numpy(a, p_list))
