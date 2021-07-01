# This script excercises many different calls to torch.nn.functional.pad
# and checks that the result matches the equivalent numpy.pad call.
import torch
import numpy as np
import itertools
from torch.testing._internal.common_utils import make_tensor

test_cases = [
    # Input size, Numpy pad mode, pad width (numpy format), kwargs numpy, kwargs torch
    ((10, 10), 'constant', ((2, 2), (2, 2)), {}, {}),
    ((10, 10), 'constant', ((2, 2), (2, 2)), {'constant_values': 10}, {'value': 10}),
    ((5, 2, 4, 1, 7), 'constant', ((1, 2), (6, 4), (10, 7), (3, 3), (0, 0)), {'constant_values': 89}, {'value': 89}),

    ((10, 10, 10), 'reflect', ((0, 0), (0, 0), (9, 9)), {}, {}),
    ((3, 3, 10, 12), 'reflect', ((0, 0), (0, 0), (2, 9), (0, 3)), {}, {}),
    ((3, 3, 10, 12), 'reflect', ((0, 0), (0, 0), (9, 9), (11, 11)), {}, {}),
    ((1, 1, 100, 100), 'reflect', ((0, 0), (0, 0), (99, 99), (99, 99)), {}, {}),
]

dtypes = [
    torch.float,
    torch.double,
    #torch.complex64
]

devices = [
    'cpu',
    'cuda'
]

# Get the equivalent
def get_mode_torch(mode_numpy):
    if mode_numpy in ['reflect', 'constant']:
        return mode_numpy
    elif mode_numpy == 'edge':
        return 'replicate'
    elif mode_numpy == 'wrap':
        return 'circular'
    else:
        assert False, f'NumPy mode "{mode_numpy}" has no PyTorch equivalent'

# Numpy's pad width arg must be a 2-D sequence of paired before- and
# after-padding for each dimension. Numpy supports other formats, but those are
# not compatible with PyTorch. PyTorch's pad width arg has its pairs in reverse
# order, and the arg is just a 1-D sequence, where each contiguous pair of
# elements makes up the before- and after-padding for one dimension. Also,
# for all modes except 'constant', we cannot include the first two dimensions
# in the PyTorch pad_width
def get_pad_width_torch(pad_width_numpy, mode_torch):
    pad_width_torch = tuple(itertools.chain.from_iterable(reversed(pad_width_numpy)))

    if mode_torch in ['reflect', 'replicate', 'circular']:
        assert all([p == 0 for p in pad_width_torch[-4:]]), f'Cannot pad first two dimensions in PyTorch with mode="{mode_torch}"'
        return pad_width_torch[:-4]

    return pad_width_torch

for device, dtype in itertools.product(devices, dtypes):
    for case_idx, (size, mode_numpy, pad_width_numpy, kwargs_numpy, kwargs_torch) in enumerate(test_cases):
        case_name = f'test_cases[{case_idx}], mode="{mode_numpy}", {device}, {str(dtype).split(".")[-1]}'

        input_torch = make_tensor(size, device, dtype, low=-9, high=9)
        input_numpy = input_torch.cpu().numpy()

        res_numpy = np.pad(input_numpy, pad_width_numpy, mode=mode_numpy, **kwargs_numpy)

        mode_torch = get_mode_torch(mode_numpy)
        pad_width_torch = get_pad_width_torch(pad_width_numpy, mode_numpy)

        res_torch = torch.nn.functional.pad(input_torch, pad_width_torch, mode=mode_torch, **kwargs_torch)

        assert torch.allclose(res_torch.cpu(), torch.from_numpy(res_numpy)), case_name

print("all test cases PASSED")

