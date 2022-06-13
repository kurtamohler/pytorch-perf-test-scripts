import torch

torch.manual_seed(0)

def my_scatter_add_(self, dim, index, src):
    assert self.dim() == index.dim()
    assert self.dim() == src.dim()
    assert dim < self.dim() and dim >= -self.dim()

    for size_idx in range(dim):
        assert index.size(size_idx) <= src.size(size_idx)

    # Wrap dim
    if dim < 0:
        dim += self.dim()

    if self.dim() == 1:
        self_flat = self
        index_flat = index
        src_flat = src 

    else:
        self_contig = self.contiguous()

        index_coords_shape = tuple(index.shape) + (self_contig.dim(),)
        index_coords = torch.empty(
            index_coords_shape,
            dtype=torch.int64,
            device=self.device)

        for dim_other in range(0, self_contig.dim()):
            if dim_other != dim:
                dim_coord_vals = torch.arange(index.shape[dim_other], device=self.device)

                for dim_unsqueeze in range(0, self_contig.dim() - 1):
                    dim_coord_vals = dim_coord_vals.unsqueeze(
                        -1 if dim_unsqueeze >= dim_other else 0)

                # The following restride and copy is the same as
                # `index_coords[..., dim_other] = dim_coord_vals`
                torch.as_strided(
                    index_coords,
                    tuple(index.shape) + (1,),
                    tuple(list(index_coords.stride())[:-1]) + (self_contig.dim(),),
                    storage_offset=dim_other
                ).copy_(dim_coord_vals.unsqueeze(-1))

        # The following restride and copy is the same as
        # `index_coords[..., dim] = index`
        torch.as_strided(
            index_coords,
            tuple(index.shape) + (1,),
            tuple(list(index_coords.stride())[:-1]) + (self_contig.dim(),),
            storage_offset=dim
        ).copy_(index.unsqueeze(-1))

        index_coords_flat = index_coords.flatten(0, -2)

        coord_strides = torch.tensor(
            self_contig.stride(),
            device=self.device
        ).unsqueeze(0).expand(index_coords_flat.shape)

        index_flat = (index_coords_flat * coord_strides).sum(dim=-1)

        self_flat = self_contig.flatten()
        
        src_reshape = torch.as_strided(
            src,
            index.shape,
            src.stride())
        src_flat = src_reshape.flatten()

    self_flat.index_put_((index_flat,), src_flat, accumulate=True)

    if not self.is_contiguous():
        self.copy_(self_flat.reshape(self.shape))

    return self

test_cases = [
    # self_size, dim, index, src

    # 1-D
    ((10,), 0, torch.tensor([0, 0, 0]), torch.tensor([1., 2., 3.])),
    ((10,), 0, torch.randint(0, 10, (1000,)), torch.randn(1000)),
    ((100,), 0, torch.randint(0, 10, (1000,)), torch.randn(1000)),

    # 2-D with dim=0
    ((10, 100), 0, torch.randint(0, 10, (20, 100)), torch.randn(20, 100)),
    ((10, 100), 0, torch.randint(0, 10, (100, 90)), torch.randn(100, 90)),
    ((100, 100), 0, torch.randint(0, 10, (100, 50)), torch.randn(100, 50)),
    ((100, 10), 0, torch.randint(0, 10, (100, 10)), torch.randn(100, 10)),
    ((100, 20), 0, torch.randint(0, 10, (100, 10)), torch.randn(100, 10)),

    ((10, 100), 0, torch.randint(0, 10, (20, 100)), torch.randn(24, 110)),
    ((3, 4), 1, torch.randint(0, 3, (3, 100)), torch.randn(4, 200)),

    # 2-D with dim=1
    ((10, 100), 1, torch.randint(0, 10, (10, 100)), torch.randn(10, 100)),
    ((20, 100), 1, torch.randint(0, 10, (10, 100)), torch.randn(10, 100)),
    ((100, 100), 1, torch.randint(0, 10, (50, 100)), torch.randn(50, 100)),
    ((100, 100), 1, torch.randint(0, 10, (100, 50)), torch.randn(100, 50)),
    ((100, 10), 1, torch.randint(0, 10, (100, 10)), torch.randn(100, 10)),
    ((100, 20), 1, torch.randint(0, 10, (100, 10)), torch.randn(100, 10)),

    # 3-D with dim=1
    ((2, 3, 4), 1, torch.tensor([
            [
                [1, 2, 0, 2],
                [0, 0, 0, 0],
            ],
            [
                [2, 2, 2, 1],
                [1, 1, 0, 0]
            ],
        ]), torch.tensor([
            [
                [1., 2, 3, 4],
                [5, 6, 7, 8],
            ],
            [
                [9, 10, 11, 12],
                [13, 14, 15, 16]
            ]
        ])),
    ((10, 20, 30, 40), 2, torch.randint(0, 30, (1, 2, 1000, 4)), torch.randn(1, 2, 1000, 4)),

    ((10, 20, 30, 40), 3, torch.randint(0, 40, (2, 2, 10, 400)), torch.randn(2, 2, 10, 400)),

]

device = 'cuda'

for self_size, dim, index, src in test_cases:
    self0 = torch.zeros(self_size).to(device)
    self1 = self0.clone()

    index = index.to(device)
    src = src.to(device)

    res0 = torch.Tensor.scatter_add(self0, dim, index, src)

    torch.use_deterministic_algorithms(True)
    my_scatter_add_(self1, dim, index, src)
    torch.use_deterministic_algorithms(False)

    assert torch.allclose(res0, self1, rtol=1e-03)

