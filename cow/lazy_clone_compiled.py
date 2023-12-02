import torch
import unittest

class MyTests(unittest.TestCase):
    def test_lazy_clone_view_0(self):
        def f(device, dtype):
            t = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)
            clone = t._lazy_clone()
            self.assertTrue(torch._C._is_cow_tensor(t))
            view = t.view([4])
            return view

        torch.compile(f)('cpu', torch.float32)

if __name__ == '__main__':
    unittest.main()
