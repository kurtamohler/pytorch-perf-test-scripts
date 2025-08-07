import torch

cases = [
    # (input, grid, kwargs)
    (
        torch.tensor([[
            [
                [[0, 1],
                [2, 3],]
            ],
            [
                [[4, 5],
                [6, 7],]
            ],
        ]], dtype=torch.float),
        torch.tensor([[[[
            [-1, -1, 0],
            [-1,  1, 0],
            [ 1, -1, 0],
            [ 1,  1, 0],
            [ 0,  0, 0],
        ]]]], dtype=torch.float),
        dict(align_corners=True)
    ),
    (
        torch.tensor([[
            [
                [0, 1],
                [2, 3],
            ],
            [
                [4, 5],
                [6, 7],
            ],
        ]], dtype=torch.half),
        torch.tensor([[[
            [-1, -1],
            [-1,  1],
            [ 1, -1],
            [ 1,  1],
            [ 0,  0],
        ]]], dtype=torch.half),
        dict(
            align_corners=True,
            interpolation_mode='bicubic',
        )
    ),
]

for input, grid, kwargs in cases:
    print('=========================')
    print(input.shape)
    for device in ['cpu', 'mps']:
        output = torch.nn.functional.grid_sample(
            input.to(device),
            grid.to(device),
            align_corners=True)
        print(output)
    print('=========================')