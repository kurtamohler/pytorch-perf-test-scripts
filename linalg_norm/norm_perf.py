import torch
import time

torch.manual_seed(0)

def measure_time(input, f, args, kwargs, num_iters):
    need_sync = input.device.type == 'cuda'

    for warmup in range(2):
        start = time.time()

        for iter in range(num_iters):
            result = f(input, *args, **kwargs)

        if need_sync:
            torch.cuda.synchronize(input.device)
        end = time.time() 

    return (end - start) / num_iters, result


def custom_norm(input, order):
    if order == 1:
        return input.abs().pow(order).sum().pow(1. / order)
    elif order == 2:
        input_abs = input.abs()
        return (input_abs * input_abs).sum().sqrt()
    elif order == float('inf'):
        return input.abs().amax()
    elif order == -float('inf'):
        return input.abs().amin()
    else:
        return input.abs().pow(order).sum().pow(1. / order)


methods = {
    'norm': torch.norm,
    'linalg_norm': torch.linalg.norm,
    'linalg_vector_norm': torch.linalg.vector_norm,
    'custom': custom_norm,
}


num_inputs = 10

print('NOTE: N/A means that the operation is not supported')
print()
print(f'dtype device order input-size {" ".join(methods.keys())}')
print()
for dtype in [torch.float, torch.cfloat, torch.double, torch.cdouble]:
    for device in ['cpu', 'cuda']:
        for order in [1, 2, 3, float('inf'), -float('inf')]:
            for input_size in [3, 8_000, 16_000, 44_100, 32 * 44_100]:
                # CPU is much slower than CUDA, so it needs fewer timed iterations
                if device == 'cpu':
                    # When tensor can no longer fit into lower cache levels,
                    # norm performance significantly drops off, so we can
                    # decrease timed iterations linearly from that point
                    if input_size >= 44_100:
                        num_iters = int(100 / (input_size / 44_100))
                        if num_iters < 10:
                            num_iters = 10
                    else:
                        num_iters = 200
                else:
                    num_iters = 500
                times_for_all_inputs = []
                for input_num in range(num_inputs):
                    input = torch.randn(input_size, dtype=dtype, device=device)
                    results = []
                    times_for_input = []
                    for method_name, f in methods.items():

                        if method_name == 'norm':
                            kwargs = {
                                'p': order
                            }
                            args = []
                        elif method_name == 'linalg_norm':
                            kwargs = {
                                'ord': order
                            }
                            args = []
                        elif method_name == 'linalg_vector_norm':
                            kwargs = {
                                'ord': order
                            }
                            args = []
                        else:
                            kwargs = {}
                            args = [order]

                        try:
                            time_seconds, result = measure_time(input, f, args, kwargs, num_iters)
                            results.append(result)
                        except RuntimeError:
                            time_seconds = 'N/A'

                        times_for_input.append(time_seconds)
                    times_for_all_inputs.append(times_for_input)
                avg_times = torch.tensor(times_for_all_inputs, dtype=torch.float64).mean(dim=0).tolist()
                # Make sure that the difference between results of the different
                # methods is very small, so we know they are all calculating the
                # same thing
                for result_ind in range(1, len(results)):
                    if results[0] != results[result_ind]:
                        assert ((results[0] - results[result_ind]) / results[0]).abs() < 0.001
                print(f'{dtype} {device} {order} {input_size} {" ".join([str(t) for t in avg_times])}')
            print()

