import torch

mode_to_int_map = {
    'sum': 0,
    'mean': 1,
    'max': 2,
}

for mode in mode_to_int_map.keys():
    print('--------------------------')
    print(mode)
    for device in ['mps', 'cpu']:
        weight = torch.tensor(
            [
                [1.0, 0, 0],
                [0, 2.0, 0],
                [0, 0, 3.0],
                [-9.3, 1., -2],
                [0, 0, 0],
            ],
            device=device,
            dtype=torch.float,
        )
        input = torch.tensor(
            [
                1, 1, 1,
                0, 1, 2, 3,
                2, 2, 2,
            ],
            device=device,
        )
        offsets = torch.tensor(
            [0, 3, 7],
            device=device,
        )
        per_sample_weights = torch.tensor(
            [
                0.1, 0.2, 0.5,
                0.11, 0.8, 2.43, 10,
                0.294, 0.1, 0.34,
            ],
            device=device,
            dtype=torch.float,
        ) if mode == 'sum' else None

        r = torch._embedding_bag(
            weight,
            input,
            offsets,
            per_sample_weights=per_sample_weights,
            mode=mode_to_int_map[mode],
            padding_idx=1)

        print(r[0])
        print(r[3])
