import torch

for mode in ['mean', 'max', 'sum']:
    print('--------------------------')
    for device in ['mps', 'cpu']:
        weight = torch.tensor(
            [
                [1.0, 0, 0],
                [0, 2.0, 0],
                [0, 0, 3.0],
                [0, 0, 0],
            ],
            device=device,
            dtype=torch.float,
        )

        input = torch.tensor(
            [
                [1, 1, 1],
                [0, 1, 2],
                [2, 2, 2],
            ],
            device=device,
        )
        per_sample_weights = torch.tensor(
            [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
            device=device,
            dtype=torch.float,
        ) if mode == 'sum' else None

        r = torch.nn.functional.embedding_bag(
            weight,
            input,
            per_sample_weights=per_sample_weights,
            mode=mode,
            padding_idx=1)

        print(r)
