# This file contains generator functions for the tensors and storages used in
# forward compatibility checking tests. The generators are supposed to return
# the same randomized data each time they are run with the same seed value.
# The generators should be run and the results saved to files in a newer
# version of PyTorch that has untyped storages. Then, an older version of
# PyTorch that has typed storages should run the generators again (making sure
# to use the same seed value that was used to initially generate them), read
# the file associated with each test case, and compare the results.

import argparse
import itertools
import os
import torch
import dill
import pickle
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

# Determine whether we have the old or new Storage API
def is_new_api():
    if not hasattr(is_new_api, 'cache'):
        try:
            torch.FloatStorage()
        except RuntimeError:
            is_new_api.cache = True
        else:
            is_new_api.cache = False

    return is_new_api.cache

def get_storage(tensor):
    if is_new_api():
        return torch.storage.TypedStorage(tensor.storage(), tensor.dtype)
    else:
        return tensor.storage()


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

        a = make_tensor((3, 2, 2), device, dtype, low=-9, high=9)
        test_cases[f'{base_name}_1'] = [
            get_storage(a),
            a.view((2, 6, 1)),
            a,
            a[1:],
        ]

    return test_cases

# Storage save/load is not fully FC


def save_cases(seed=0, root='pickles'):
    print('-----------------')
    print('Saving test cases')
    print('-----------------')
    if not os.path.exists(root):
        os.makedirs(root)

    for case_name, save_list in regular_serialization(seed).items():
        pickle_settings = itertools.product(
            # new zip format
            [True, False],
            # protocol number
            [0, 1, 2, 3, 4, 5],
            # pickle module
            [dill, pickle],
            # use pickle module directly, rather than torch.load/save
            [True, False])

        for use_new_zip, proto, module, direct in pickle_settings:
            file_name = f'{case_name}_{"newzip" if use_new_zip else "oldzip"}_proto{proto}_{module.__name__}'

            if direct:
                if use_new_zip:
                    continue

                file_name += '_direct'
                with open (os.path.join(root, file_name), 'wb') as f:
                    module.dump(
                        save_list,
                        f,
                        protocol=proto)

            else:
                torch.save(
                    save_list,
                    os.path.join(root, file_name),
                    _use_new_zipfile_serialization=use_new_zip,
                    pickle_protocol=proto,
                    pickle_module=module)


def load_and_check_cases(seed=0, root='pickles'):
    print('-----------------------------------------')
    print('Load test cases and check for correctness')
    print('-----------------------------------------')
    for case_name, check_list in regular_serialization(seed).items():
        pickle_settings = itertools.product(
            # new zip format
            [True, False],
            # protocol number
            [0, 1, 2, 3, 4, 5],
            # pickle module
            [dill, pickle],
            # use pickle module directly, rather than torch.load/save
            [True, False])

        for use_new_zip, proto, module, direct in pickle_settings:
            # TODO: Protocol 0 doesn't work, even if we save and load from the
            # same pytorch version. Probably should file an issue
            if proto == 0:
                continue

            file_name = f'{case_name}_{"newzip" if use_new_zip else "oldzip"}_proto{proto}_{module.__name__}'

            if direct:
                if use_new_zip:
                    continue
                file_name += '_direct'
                print(file_name)
                with open(os.path.join(root, file_name), 'rb') as f:
                    loaded_list = module.load(
                        f)
            else:
                print(file_name)
                loaded_list = torch.load(
                    os.path.join(root, file_name),
                    pickle_module=module)

            for idx0 in range(len(check_list)):
                check_val0 = check_list[idx0]
                loaded_val0 = loaded_list[idx0]


                # Check that loaded values are what they should be
                assert type(check_val0) == type(loaded_val0)

                if torch.is_tensor(check_val0):
                    assert check_val0.eq(loaded_val0).all()
                elif torch.is_storage(check_val0):
                    assert (check_val0.tolist() == loaded_val0.tolist())
                elif isinstance(check_val0, torch.storage.TypedStorage):
                    assert check_val0.dtype == loaded_val0.dtype
                    assert check_val0.storage.tolist()== loaded_val0.storage.tolist()


                # Check that storage sharing is preserved
                for idx1 in range(len(check_list) - 1 - idx0):
                    check_val1 = check_list[idx1]
                    loaded_val1 = loaded_list[idx0]

                    if check_val0.data_ptr() == check_val1.data_ptr():
                        assert loaded_val0.data_ptr() == loaded_val1.data_ptr()

    print('all cases PASSED')


if __name__ == '__main__':
    seed = 0

    parser = argparse.ArgumentParser(
        description='Test FC for PyTorch serialization')

    parser.add_argument(
        '--save',
        action='store_true',
        default=False,
        help='Save all test cases')

    parser.add_argument(
        '--load',
        action='store_true',
        default=False,
        help='Load all test cases and check for correctness')

    args = parser.parse_args()

    if args.save:
        save_cases(seed)

    if args.load:
        load_and_check_cases(seed)
