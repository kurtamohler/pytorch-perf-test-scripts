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

# Generate test cases for regular serialization (like torch.save/load)
def regular_serialization(seed=0):
    torch.manual_seed(seed)
    test_cases = {}
    for dtype, device in itertools.product(all_dtypes, all_devices):
        base_name = f'regular_serialization_{dtype_name(dtype)}_{device}'

        test_cases[f'{base_name}_0'] = [
            make_tensor((3, 5), device, dtype, low=-9, high=9)
        ]

        a = make_tensor((15, 5, 5), device, dtype, low=-9, high=9)
        test_cases[f'{base_name}_1'] = [
            get_storage(a),
            a.view((5, 3, 25)),
            a,
            a[1:],
        ]

        if dtype in [torch.float32, torch.float64]:
            m = torch.nn.Linear(5, 10)
            m.weight = torch.nn.Parameter(m.weight.to(dtype).to(device))
            m.bias = torch.nn.Parameter(m.bias.to(dtype).to(device))
            test_cases[f'{base_name}_2'] = [
                m
            ]


        # TODO: test JIT
        # TODO: test quantized (per_tensor_affine, per_channel_affine, and per_channel_affine_float_qparams

        # TODO: test sparse COO

    return test_cases

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
            # use pickle module dump/load, rather than torch.save/load
            [True, False])

        for use_new_zip, proto, module, pickledump in pickle_settings:
            file_name = f'{case_name}_{"newzip" if use_new_zip else "oldzip"}_proto{proto}_{module.__name__}'

            if pickledump:
                if use_new_zip:
                    continue

                file_name += '_pickledump'
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
            # use pickle module dump/load, rather than torch.save/load
            [True, False])

        for use_new_zip, proto, module, pickledump in pickle_settings:
            # TODO: Protocol 0 doesn't work, even if we save and load from the
            # same pytorch version. Probably should file an issue
            if proto == 0:
                continue

            file_name = f'{case_name}_{"newzip" if use_new_zip else "oldzip"}_proto{proto}_{module.__name__}'

            if pickledump:
                if use_new_zip:
                    continue
                file_name += '_pickledump'
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
                    assert check_val0.device == loaded_val0.device
                    assert check_val0.eq(loaded_val0).all()
                elif torch.is_storage(check_val0):
                    assert check_val0.device == loaded_val0.device
                    assert check_val0.tolist() == loaded_val0.tolist()
                elif isinstance(check_val0, torch.storage.TypedStorage):
                    assert check_val0.storage.device == loaded_val0.storage.device
                    assert check_val0.dtype == loaded_val0.dtype
                    assert check_val0.tolist()== loaded_val0.tolist()
                elif issubclass(type(check_val0), torch.nn.Module):
                    param_pairs = zip(check_val0.parameters(), loaded_val0.parameters())
                    assert all([p0.device == p1.device for p0, p1 in param_pairs])
                    assert all([p0.eq(p1).all() for p0, p1 in param_pairs])

                # Check that storage sharing is preserved
                for idx1 in range(len(check_list) - 1 - idx0):
                    check_val1 = check_list[idx1]
                    loaded_val1 = loaded_list[idx0]

                    if check_val0.data_ptr() == check_val1.data_ptr():
                        assert loaded_val0.data_ptr() == loaded_val1.data_ptr()

    print('all cases PASSED')


if __name__ == '__main__':
    seed = 0

    parser = argparse.ArgumentParser(description=(
        'Test FC and BC for PyTorch serialization. To use this test, run with '
        'the "save" option in one environment to create pickle files for many '
        'different tensors/storages. Then run with the "load" option in '
        'a different environment to check that the correct data gets loaded back '
        'up.'))

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

    parser.add_argument(
        '--seed',
        default=0,
        type=int,
        metavar='S',
        help='Set PyTorch RNG seed')

    args = parser.parse_args()

    if args.save:
        save_cases(args.seed)

    if args.load:
        load_and_check_cases(args.seed)

    if not args.save and not args.load:
        parser.print_help()
