import torch
import numpy
import math

def gram_matrix(a):
    if a.size()[-1] <= a.size()[-2]:
        return a.transpose(-1, -2).conj().matmul(a)
    elif a.size()[-1] > a.size()[-2]:
        return a.matmul(a.transpose(-1, -2).conj())

def power_iteration(a, max_iters: int):
    eigenvector = torch.randn(a.size()[-1])
    eigenvalue = eigenvector.norm()
    eigenvector /= eigenvalue

    # Iterate to approximate the dominant eigenvector
    for _ in range(max_iters):
        eigenvector = a.matmul(eigenvector)
        eigenvalue = eigenvector.norm()
        eigenvector /= eigenvalue

    return eigenvalue

# Based on https://math.stackexchange.com/questions/1255709/finding-the-largest-singular-value-easily
def approx_eigenvalue(a, max_iters: int, mode: str):
    # From experiments, using the Gram matrix even if `a` is already square
    # gives more accuracte results by several orders of magnitude. I'm not sure
    # why this is, but I suspect it has something to do with the fact that the
    # Gram matrix is symmetrical
    if a.size()[-1] <= a.size()[-2]:
        a_gram = a.transpose(-1, -2).conj().matmul(a)
    elif a.size()[-1] > a.size()[-2]:
        a_gram = a.matmul(a.transpose(-1, -2).conj())

    if mode == 'max':
        eigenvalue = power_iteration(a_gram, max_iters)
    elif mode == 'min':
        eigenvalue = (1 / power_iteration(a_gram.inverse(), max_iters))
    else:
        raise ValueError("'mode' must be either 'max' or 'min', not '%s'" % mode)

    eigenvalue = eigenvalue.sqrt()

    return eigenvalue

def approx_spectral_largest(a, iters: int, method: str):
    class MyModule(torch.nn.Module):
        def __init__(self, weight):
            super(MyModule, self).__init__()
            self.weight = torch.nn.Parameter(weight)
                
        def forward(self):
            return self.weight
                
    b = torch.nn.utils.spectral_norm(
        MyModule(a),
        n_power_iterations=iters)()
            
    
    if method == 'median':
        return (a / b).median()
    elif method == 'mean':
        return (a / b).mean()
    else:
        raise RuntimeError('method "%s" not available' % method)

if __name__ == '__main__':
    verbose = False
    
    samples_per_size = 1000
    
    for norm_method in ['spectral_norm_mean', 'spectral_norm_median', 'Custom power iteration', 'numpy.linalg.norm']:
        print('norm_method: %s' % norm_method)
        print('input_mat_size, num_power_iters, smallest_eigenvalue_relative_error_mean+/-std, largest_eigenvalue_relative_error_mean+/-std')
        for size in [
                (1, 100),
                (5, 100),
                (10, 100),
                (25, 100),
                (50, 100),
                (75, 100),
                (100, 100),
                (100, 75),
                (100, 50),
                (100, 25),
                (100, 10),
                (100, 5),
                (100, 1),
            ]:
            max_power_iters = 1 if min(*size) == 1 else 20
        
            smallest_rel_errors = []
            largest_rel_errors = []
        
            for _ in range(samples_per_size):
                a = (torch.rand(*size) - 0.5) * 2
                eigenvalues = a.svd(True, False)[1].abs()
                #eigenvalues = torch.tensor(numpy.linalg.svd(a.cpu().numpy())[1], device=a.device).abs()
                
                exact_largest = eigenvalues.max()
                exact_smallest = eigenvalues.min()
                
                if norm_method == 'numpy.linalg.norm':
                    approx_smallest = numpy.linalg.norm(a.cpu(), ord=-2)
                    approx_largest = numpy.linalg.norm(a.cpu(), ord=2)
                elif norm_method == 'Custom power iteration':
                    approx_largest = approx_eigenvalue(a, max_power_iters, 'max')
                    approx_smallest = approx_eigenvalue(a, max_power_iters, 'min')
                elif norm_method == 'spectral_norm_mean':
                    approx_largest = approx_spectral_largest(a, max_power_iters, 'mean')
                    approx_smallest = 0
                elif norm_method == 'spectral_norm_median':
                    approx_largest = approx_spectral_largest(a, max_power_iters, 'median')
                    approx_smallest = 0
                else:
                    raise RuntimeError('Unknown norm method: %s' % norm_method)
        
                error_smallest = ((approx_smallest - exact_smallest) / exact_smallest).abs()
                error_largest = ((approx_largest - exact_largest) / exact_largest).abs()
        
                smallest_rel_errors.append(error_smallest)
                largest_rel_errors.append(error_largest)
        
            smallest_val_error_std, smallest_val_error_mean = torch.std_mean(torch.tensor(smallest_rel_errors))
            largest_val_error_std, largest_val_error_mean = torch.std_mean(torch.tensor(largest_rel_errors))
        
            print("%s, %d, %f+/-%f, %f+/-%f" % (size, max_power_iters, smallest_val_error_mean, smallest_val_error_std, largest_val_error_mean, largest_val_error_std))
        print()
