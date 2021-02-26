import torch
import random
from itertools import product
from math import ceil
from time import time

test_cases = product(
    # Devices
    ['cuda'],

    # Include backward
    [False, True],

    # dtype for the embedding
    [torch.float, torch.double],

    # Mode
    ['sum', 'mean', 'max'],
    #['max'],

    # Embedding sizes
    [(100, 100)],

    # Indices sizes
    [(1000, 1000)],

    # Padding ratio
    [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
)

# Each case will be run multiple times with different inputs, and the median
# time will be reported
num_inputs_per_case = 10

# Number of times to call embedding_bag within a timed loop
timed_iters = 100

def measure_run_time(indices, embedding, mode, padding_idx, include_backward, timed_iters):
    need_sync = embedding.device.type == 'cuda'

    # If we're including backward function in the measurement, generate an
    # output gradient to use for all backward calls. This minimizes overhead
    # compared to doing something like `result.sum().backward()`.
    if include_backward:
        output_grad = torch.ones(indices.size(0), embedding.size(1), device=embedding.device, dtype=embedding.dtype)

    try:
        for warmup in range(2):
            start = time()

            for iter in range(timed_iters):
                result = torch.nn.functional.embedding_bag(
                    indices,
                    embedding,
                    mode=mode,
                    padding_idx=padding_idx)

                if include_backward:
                    result.backward(output_grad)


            if need_sync:
                torch.cuda.synchronize(embedding.device)

            end = time()
        return (end - start) / timed_iters
    except TypeError:
        # If padding_idx is not supported, only run padding_idx=None case
        if padding_idx is not None:
            return 0

        for warmup in range(2):
            start = time()

            for iter in range(timed_iters):
                result = torch.nn.functional.embedding_bag(
                    indices,
                    embedding,
                    mode=mode)

                if include_backward:
                    result.backward(output_grad)

            if need_sync:
                torch.cuda.synchronize(embedding.device)

            end = time()
        return (end - start) / timed_iters

print((
    f'| device | include_backward | dtype | mode | '
    f'embedding_size | indices_size | padding_ratio | '
    f'median_time |'))
print(('| --- ' * 7) + '|')

for device, include_backward, dtype, mode, embedding_size, indices_size, padding_ratio in test_cases:
    assert padding_ratio >= 0.0 and padding_ratio <= 1.0

    num_bags = indices_size[0]
    num_embeddings = embedding_size[0]

    # Use the first embedding index as padding. Randomizing the padding index
    # might give a bit better performance coverage, but this is simpler
    padding_idx = 0 if (padding_ratio > 0) else None

    run_times = []

    # Seed RNG so that each test case uses comparable inputs
    torch.manual_seed(0)
    random.seed(0)

    for input_num in range(num_inputs_per_case):
        embedding = torch.randn(
            embedding_size,
            dtype=dtype,
            device=device,
            requires_grad=include_backward)

        # If using padding, first avoid setting anything to padding_idx,
        # then fill in the required number of padding indices randomly
        indices = torch.randint(
            0 if (padding_idx is None) else 1,
            num_embeddings,
            indices_size,
            device=device)

        if padding_idx is not None:
            num_padding = ceil(padding_ratio * indices.numel())
            indices_to_pad = random.sample(range(indices.numel()), num_padding)
            indices.view(-1).index_fill_(
                0,
                torch.tensor(indices_to_pad, device=device),
                padding_idx)

            # double check that we have the right amount of padding
            num_padding_check = indices.eq(padding_idx).sum()
            assert num_padding_check == num_padding

        run_times.append(measure_run_time(
            indices, embedding, mode, padding_idx, include_backward, timed_iters))

    median_time = torch.tensor(run_times).median()

    if median_time != 0:
        print((
            f'| {device} | {include_backward} | {dtype} | {mode} | '
            f'{embedding_size} | {indices_size} | {padding_ratio} | '
            f'{median_time} |'))

