import torch

mode_map = {'sum': 0, 'mean': 1, 'max': 2}

num_words = 4
feature_size = 4


for device in ['cpu', 'mps']:
    input = torch.tensor([3, 1, 2, 2, 0, 2], device=device)
    offsets = torch.tensor([0, 3], device=device)
    weights = torch.tensor([
        [float('nan'), 1, 1, 1],
        [1, float('nan'), 1, 1],
        [1, 1, 1, 1],
        [float('nan'), 1, 1, 1],
    ], dtype=torch.float, device=device)

    r, _, _, i = torch._embedding_bag(
        weights,
        input,
        offsets,
        #include_last_offset=True,
        padding_idx=-1,
        mode=mode_map['max'],
    )

    print(r)
    print(i)
