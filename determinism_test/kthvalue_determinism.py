import torch

# Runs kthvalue() multiple times on a tensor with the specified number of
# duplicate values. K is varied and the entire experiment is repeated
# several times to check whether each combination of arguments given to
# kthvalue() ever results in a different index output value
def test_kthvalue(size, num_copies, device):
    result_indices_record = []

    for trial_num in range(10):
        # Seed RNG to repeat the same experiment multiple times
        torch.manual_seed(size * num_copies)

        a = torch.rand(size).to(device)

        # Copy a random element num_copies-1 times
        source_index = torch.randint(size, (1,)).item()
        dest_indices = []
        for _ in range(num_copies-1):
            index = source_index
            while index == source_index or index in dest_indices:
                index = torch.randint(size, (1,)).item()
            dest_indices.append(index)
            a[index] = a[source_index]

        # Need to find out the k values of the repeated elements, by sorting
        # the tensor and finding the indices of the elements in the sorted list
        k_list = (torch.nonzero(a.sort()[0].eq(a[source_index])) + 1).flatten().tolist()

        for k_ind, k in enumerate(k_list):
            result_index = None
            for determinism_check_num in range(10):
                new_result_index = a.kthvalue(k)[1].item()
                if result_index is not None:
                    if new_result_index != result_index:
                        return False, result_indices_record
                result_index = new_result_index

            if trial_num == 0:
                result_indices_record.append(result_index)
            else:
                if result_indices_record[k_ind] != result_index:
                    return False, result_indices_record

    return True, result_indices_record

num_copies = 11

print('device size number-of-copies k-to-index-unique-mapping maybe-deterministic')
for device in ['cpu', 'cuda']:
    for size in [20, 100, 1_000, 10_000]:#, 10_000_000]:
        for num_copies in [1, 2, 5, 10, 11]:
            is_deterministic, result_indices = test_kthvalue(size, num_copies, device)
            if len(result_indices):
                has_unique_map = len(result_indices) == len(set(result_indices))
            else:
                has_duplicates = 'N/A'

            print(f'{device} {size} {num_copies} {has_unique_map} {is_deterministic}')
