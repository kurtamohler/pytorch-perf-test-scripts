import torch
import time

def measure_time(input, f, args, kwargs, num_iters):
    for warmup in range(2):
        start = time.time()

        for iter in range(num_iters):
            result = f(input, *args, **kwargs)

        end = time.time() 

    return (end - start) / num_iters, result


methods = {
    'norm': torch.norm,
    'linalg_norm': torch.linalg.norm,
    'custom-order-1': lambda input, order: input.abs().pow(order).sum().pow(1. / order),
    'custom': lambda input, order: input.abs().pow(order).sum().pow(1. / order)
}


num_iters = 10000

print('NOTE: N/A means that the operation is not supported')
print()
print('dtype device order input-size method time-seconds')
for dtype in [torch.float, torch.cfloat, torch.double, torch.cdouble]:
    for device in ['cpu', 'cuda']:
        for order in [1, 2, 3]:
            for input_size in [3, 8_000, 16_000, 44_100, 32 * 44_100]:
                input = torch.randn(input_size, dtype=dtype, device=device)
                results = []
                for method_name, f in methods.items():
                    if method_name == 'custom' and order == 1:
                        continue
                    elif method_name == 'custom-order-1' and order != 1:
                        continue

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
                    else:
                        kwargs = {}
                        args = [order]

                    try:
                        time_seconds, result = measure_time(input, f, args, kwargs, 100)
                        results.append(result)
                    except RuntimeError:
                        #print(f'{device} {input_size} {order} {method_name} N/A')
                        #continue
                        time_seconds = 'N/A'

                    print(f'{dtype} {device} {order} {input_size} {method_name} {time_seconds}')
                # Make sure that the difference between results of the different
                # methods is very small, so we know they are all calculating the
                # same thing
                for result_ind in range(1, len(results)):
                    if results[0] != results[result_ind]:
                        assert ((results[0] - results[result_ind]) / results[0]).abs() < 0.001
            print()







