import torch
import torch.distributed.run
import io
import subprocess

with open('distributed_test0.py', 'w') as f:
    f.write('import torch\nprint("hello there")\n')

processes = []

processes.append(subprocess.Popen([
    'torchrun',
    '--nnodes', '2',
    '--nproc_per_node', '2',
    '--node_rank', '0',
    'distributed_test0.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE))

processes.append(subprocess.Popen([
    'torchrun',
    '--nnodes', '2',
    '--nproc_per_node', '2',
    '--node_rank', '1',
    'distributed_test0.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE))

for rank, process in enumerate(processes):
    process_out, process_err = process.communicate()

    print('=========================')
    print(f'rank: {rank}')
    print('stdout:')
    print(process_out)
    print('stderr:')
    print(process_err)


print('done')
