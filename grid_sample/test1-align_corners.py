import torch

a = torch.tensor(
    [[[[1, 2, 3, 5, 6, 7, 8]]]],
    dtype=torch.float)

grid = torch.tensor(
    [[[
        [-1, 0],
        [ 0, 0],
        [ 1, 0],
    ]]],
    dtype=torch.float)

for align_corners in [True, False]:
    print('========================')
    print(f'{align_corners=}')

    r = torch.nn.functional.grid_sample(
        a,
        grid,
        align_corners=align_corners,
        padding_mode='border',
    )

    print(r)
