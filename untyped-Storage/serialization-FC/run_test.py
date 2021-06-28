# This file contains generator functions for the tensors and storages used in
# forward compatibility checking tests. The generators are supposed to return
# the same randomized data each time they are run with the same seed value.
# The generators should be run and the results saved to files in a newer
# version of PyTorch that has untyped storages. Then, an older version of
# PyTorch that has typed storages should run the generators again (making sure
# to use the same seed value that was used to initially generate them), read
# the file associated with each test case, and compare the results.

import itertools
import os
import torch
from torch.testing._internal.common_utils import make_tensor

all_dtypes = [
    torch.int32,
    torch.int64,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
]

all_devices = [
    'cpu',
    'cuda',
]

def dtype_name(dtype):
    return str(dtype).split('.')[-1]

# ==========================
# Test regular serialization with tensors
#
# Tensor save/load should be mostly FC. However, if two tensors of different
# dtypes share a storage, saving them is not FC.
def regular_serialization(seed=0):
    torch.manual_seed(seed)
    test_cases = {}
    for dtype, device in itertools.product(all_dtypes, all_devices):
        base_name = f'regular_serialization_{dtype_name(dtype)}_{device}'
        test_cases[f'{base_name}_0'] = [
            make_tensor((3, 5), device, dtype, low=-9, high=9)
        ]

    return test_cases

# Storage save/load is not fully FC


def pickle_all(seed=0, root='pickles'):
    print('Pickling for all test cases')
    if not os.path.exists(root):
        os.makedirs(root)

    for case_name, save_list in regular_serialization(seed).items():
        for use_new_zip, proto in itertools.product([True, False], [0, 1, 2, 3, 4, 5]):
            file_name = f'{case_name}_{"newzip" if use_new_zip else "oldzip"}_proto{proto}'
            torch.save(
                save_list,
                os.path.join(root, file_name),
                _use_new_zipfile_serialization=use_new_zip,
                pickle_protocol=proto)


def unpickle_all(seed=0, root='pickles'):
    print('Unpickling for all test cases')
    passed = True
    for case_name, check_list in regular_serialization(seed).items():
        for use_new_zip, proto in itertools.product([True, False], [0, 1, 2, 3, 4, 5]):
            # TODO: Protocol 0 doesn't work, even if we save and load from the
            # same pytorch version. Probably should file an issue
            if proto == 0:
                continue

            file_name = f'{case_name}_{"newzip" if use_new_zip else "oldzip"}_proto{proto}'
            loaded_list = torch.load(
                os.path.join(root, file_name))

            for check_val, loaded_val in zip(check_list, loaded_list):
                if not check_val.eq(check_val).all():
                    print(f'{file_name}: FAIL')
                    passed = False

    if passed:
        print('all cases PASSED')


if __name__ == '__main__':
    seed = 0

    # Determine whether we have the old or new Storage API
    try:
        torch.FloatStorage()
    except RuntimeError:
        is_new_api = True
    else:
        is_new_api = False

    if is_new_api:
        pickle_all(seed)
    else:
        unpickle_all(seed)
