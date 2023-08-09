import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def run(rank, size, a, backend='gloo'):
    time.sleep(3)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

    dist.barrier()
    print('==================')
    print(f'rank {rank}, a.data_ptr(): {a.data_ptr()}')

    dist.barrier()
    print('==================')
    print('create other storage')
    other_storage = torch.UntypedStorage(10)

    dist.barrier()
    if rank == 0:
        print('==================')
        print('rank 0 get storage')
        s = a.untyped_storage()

    dist.barrier()
    if rank == 1:
        print('==================')
        print('rank 1 get storage')
        print(a)
        s = a.untyped_storage()

    dist.barrier()
    if rank == 0:
        print('==================')
        print("rank 0 replace a's storage")
        a.set_(other_storage)
        print('==================')
        print('rank 0 del storage')
        del s

    dist.barrier()
    if rank == 1:
        print('==================')
        print("rank 1 replace a's storage")
        a.set_(other_storage)
        print('==================')
        print('rank 1 del storage')
        del s

    dist.barrier()
    if rank == 0:
        print('==================')
        print('clean up')
    dist.barrier()
    del other_storage
    del a

if __name__ == "__main__":
    a = torch.arange(10)
    a.share_memory_()
    print(f'parent, a.data_ptr(): {a.data_ptr()}')
    size = 2 
    processes = []
    for rank in range(size):
        p = mp.Process(target=run, args=(rank, size, a))
        processes.append(p)

    print('======================')
    print('del tensor from parent process')
    del a
    print('======================')
    print('starting child processes')

    for p in processes:
        p.start()

    for p in processes:
        p.join()
