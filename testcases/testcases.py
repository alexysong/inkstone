import unittest
import numpy as np
import torch
from inkstone.backends.NumpyBackend import NumpyBackend
from inkstone.backends.TorchBackend import TorchBackend

class TestBackendComparison(unittest.TestCase):
    def setUp(self):
        self.np_backend = NumpyBackend()
        self.torch_backend = TorchBackend()
        np.random.seed(42)
        torch.manual_seed(42)

    def assert_close(self, np_result, torch_result, rtol=1e-5, atol=1e-8):
        if isinstance(np_result, np.ndarray) and isinstance(torch_result, torch.Tensor):
            np.testing.assert_allclose(np_result, torch_result.cpu().numpy(), rtol=rtol, atol=atol)
        else:
            self.assertAlmostEqual(np_result, torch_result.item(), delta=atol)

    def test_abs(self):
        x = np.random.randn(5, 5)
        np_result = self.np_backend.abs(x)
        torch_result = self.torch_backend.abs(torch.from_numpy(x))
        self.assert_close(np_result, torch_result)

    def test_sqrt(self):
        x = np.abs(np.random.randn(5, 5))
        np_result = self.np_backend.sqrt(x)
        torch_result = self.torch_backend.sqrt(torch.from_numpy(x))
        self.assert_close(np_result, torch_result)

    def test_arange(self):
        np_result = self.np_backend.arange(10)
        torch_result = self.torch_backend.arange(10)
        self.assert_close(np_result, torch_result)

    def test_ceil(self):
        x = np.random.rand(5, 5)
        np_result = self.np_backend.ceil(x)
        torch_result = self.torch_backend.ceil(torch.from_numpy(x))
        self.assert_close(np_result, torch_result)

    def test_where(self):
        condition = np.random.rand(5, 5) > 0.5
        x = np.random.randn(5, 5)
        y = np.random.randn(5, 5)
        np_result = self.np_backend.where(condition, x, y)
        torch_result = self.torch_backend.where(torch.from_numpy(condition), torch.from_numpy(x), torch.from_numpy(y))
        self.assert_close(np_result, torch_result)

    def test_diag(self):
        x = np.random.randn(5)
        np_result = self.np_backend.diag(x)
        torch_result = self.torch_backend.diag(torch.from_numpy(x))
        self.assert_close(np_result, torch_result)

    def test_sin_cos(self):
        x = np.random.randn(5, 5)
        for func in ['sin', 'cos']:
            np_result = getattr(self.np_backend, func)(x)
            torch_result = getattr(self.torch_backend, func)(torch.from_numpy(x))
            self.assert_close(np_result, torch_result)

    def test_arcsin_arccos(self):
        x = np.random.uniform(-1, 1, (5, 5))
        for func in ['arcsin', 'arccos']:
            np_result = getattr(self.np_backend, func)(x)
            torch_result = getattr(self.torch_backend, func)(torch.from_numpy(x))
            self.assert_close(np_result, torch_result)

    def test_ones_zeros(self):
        for func in ['ones', 'zeros']:
            np_result = getattr(self.np_backend, func)((3, 3))
            torch_result = getattr(self.torch_backend, func)((3, 3))
            self.assert_close(np_result, torch_result)

    def test_square(self):
        x = np.random.randn(5, 5)
        np_result = self.np_backend.square(x)
        torch_result = self.torch_backend.square(torch.from_numpy(x))
        self.assert_close(np_result, torch_result)

    def test_concatenate(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 3)
        np_result = self.np_backend.concatenate([x, y], axis=0)
        torch_result = self.torch_backend.concatenate([torch.from_numpy(x), torch.from_numpy(y)], dim=0)
        self.assert_close(np_result, torch_result)

    def test_exp(self):
        x = np.random.randn(5, 5)
        np_result = self.np_backend.exp(x)
        torch_result = self.torch_backend.exp(torch.from_numpy(x))
        self.assert_close(np_result, torch_result)

    def test_sinc(self):
        x = np.random.randn(5, 5)
        np_result = self.np_backend.sinc(x)
        torch_result = self.torch_backend.sinc(torch.from_numpy(x))
        self.assert_close(np_result, torch_result)

    def test_tan(self):
        x = np.random.randn(5, 5)
        np_result = self.np_backend.tan(x)
        torch_result = self.torch_backend.tan(torch.from_numpy(x))
        self.assert_close(np_result, torch_result)

    def test_roll(self):
        x = np.random.randn(5, 5)
        np_result = self.np_backend.roll(x, shift=2, axis=0)
        torch_result = self.torch_backend.roll(torch.from_numpy(x), shifts=2, dims=0)
        self.assert_close(np_result, torch_result)

    def test_sum(self):
        x = np.random.randn(5, 5)
        np_result = self.np_backend.sum(x, axis=0)
        torch_result = self.torch_backend.sum(torch.from_numpy(x), dim=0)
        self.assert_close(np_result, torch_result)

    def test_dot(self):
        x = np.random.randn(5)
        y = np.random.randn(5)
        torch_result = self.torch_backend.dot(torch.from_numpy(x), torch.from_numpy(y))
        np_result = self.np_backend.dot(x, y)

        self.assert_close(np_result, torch_result)

    def test_hsplit(self):
        x = np.random.randn(4, 4)
        np_result = self.np_backend.hsplit(x, 2)
        torch_result = self.torch_backend.hsplit(torch.from_numpy(x), 2)
        for np_part, torch_part in zip(np_result, torch_result):
            self.assert_close(np_part, torch_part)

    def test_repeat(self):
        x = np.random.randn(2, 2)
        np_result = self.np_backend.repeat(x, 3, axis=0)
        torch_result = self.torch_backend.repeat(torch.from_numpy(x), 3, dim=0)
        self.assert_close(np_result, torch_result)

    def test_reshape(self):
        x = np.random.randn(4, 4)
        np_result = self.np_backend.reshape(x, (2, 8))
        torch_result = self.torch_backend.reshape(torch.from_numpy(x), (2, 8))
        self.assert_close(np_result, torch_result)

    def test_moveaxis(self):
        x = np.random.randn(2, 3, 4)
        np_result = self.np_backend.moveaxis(x, 0, -1)
        torch_result = self.torch_backend.moveaxis(torch.from_numpy(x), 0, -1)
        self.assert_close(np_result, torch_result)

    def test_full(self):
        np_result = self.np_backend.full((3, 3), 5.0)
        torch_result = self.torch_backend.full((3, 3), 5.0)
        self.assert_close(np_result, torch_result)

    def test_logical_not(self):
        x = np.random.rand(5, 5) > 0.5
        np_result = self.np_backend.logical_not(x)
        torch_result = self.torch_backend.logical_not(torch.from_numpy(x))
        self.assert_close(np_result, torch_result)

    def test_maximum(self):
        x = np.random.randn(5, 5)
        y = np.random.randn(5, 5)
        np_result = self.np_backend.maximum(x, y)
        torch_result = self.torch_backend.maximum(torch.from_numpy(x), torch.from_numpy(y))
        self.assert_close(np_result, torch_result)

    def test_einsum(self):
        x = np.random.randn(3, 4)
        y = np.random.randn(4, 5)
        np_result = self.np_backend.einsum('ij,jk->ik', x, y)
        torch_result = self.torch_backend.einsum('ij,jk->ik', torch.from_numpy(x), torch.from_numpy(y))
        self.assert_close(np_result, torch_result)

    def test_lu_factor(self):
        A = np.random.randn(5, 5)
        np_LU, np_pivots = self.np_backend.lu_factor(A)
        torch_LU, torch_pivots = self.torch_backend.lu_factor(torch.from_numpy(A))
        self.assert_close(np_LU, torch_LU)
        self.assert_close(np_pivots, torch_pivots - 1)  # Adjust for 0-based indexing

    def test_fft_ifftshift(self):
        x = np.random.randn(8)

        # NumPy backend
        np_result = self.np_backend.fft.ifftshift(x)

        # PyTorch backend
        torch_x = torch.from_numpy(x)
        torch_result = self.torch_backend.fft.ifftshift(torch_x)

        self.assert_close(np_result, torch_result)

    def test_slogdet(self):
        A = np.random.randn(5, 5)
        np_sign, np_logdet = self.np_backend.slogdet(A)
        torch_sign, torch_logdet = self.torch_backend.slogdet(torch.from_numpy(A))
        self.assert_close(np_sign, torch_sign)
        self.assert_close(np_logdet, torch_logdet)

    def test_solve(self):
        A = np.random.randn(5, 5)
        b = np.random.randn(5)
        np_result = self.np_backend.solve(A, b)
        torch_result = self.torch_backend.solve(torch.from_numpy(A), torch.from_numpy(b))
        self.assert_close(np_result, torch_result)

    def test_eye(self):
        np_result = self.np_backend.eye(5)
        torch_result = self.torch_backend.eye(5)
        self.assert_close(np_result, torch_result)

    def test_conj(self):
        x = np.random.randn(5) + 1j * np.random.randn(5)
        np_result = self.np_backend.conj(x)
        torch_result = torch.conj_physical(torch.from_numpy(x))
        self.assert_close(np_result, torch_result)

    def test_cross(self):
        x = np.random.randn(3)
        y = np.random.randn(3)
        np_result = self.np_backend.cross(x, y)
        torch_result = self.torch_backend.cross(torch.from_numpy(x), torch.from_numpy(y))
        self.assert_close(np_result, torch_result)

    def test_linspace(self):
        np_result = self.np_backend.linspace(0, 10, 5)
        torch_result = self.torch_backend.linspace(0, 10, 5)
        self.assert_close(np_result, torch_result)

    def test_pi(self):
        self.assertAlmostEqual(self.np_backend.pi, self.torch_backend.pi, places=7)

    def test_castType(self):
        x = np.random.randn(5)
        np_result = self.np_backend.castType(x, self.np_backend.float64)
        torch_result = self.torch_backend.castType(torch.from_numpy(x), self.torch_backend.float64)
        self.assert_close(np_result, torch_result)


    def test_meshgrid(self):
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 2)
        np_result = self.np_backend.meshgrid(x, y)
        torch_result = self.torch_backend.meshgrid(torch.from_numpy(x), torch.from_numpy(y))
        for np_grid, torch_grid in zip(np_result, torch_result):
            self.assert_close(np_grid, torch_grid)

    def test_getSize(self):
        x = np.random.randn(3, 4, 5)
        np_result = self.np_backend.getSize(x)
        torch_result = self.torch_backend.getSize(torch.from_numpy(x))
        self.assertEqual(np_result, torch_result)

    def test_clone(self):
        x = np.random.randn(5, 5)
        np_result = self.np_backend.clone(x)
        torch_result = self.torch_backend.clone(torch.from_numpy(x))
        self.assert_close(np_result, torch_result)
        self.assertFalse(np.may_share_memory(x, np_result))

    def test_triu_indices(self):
        np_result = self.np_backend.triu_indices(5, 5, 1)
        torch_result = self.torch_backend.triu_indices(5, 5, 1)
        self.assert_close(np_result[0], torch_result[0])
        self.assert_close(np_result[1], torch_result[1])

    def test_lu_solve(self):
        A = np.random.randn(5, 5)
        b = np.random.randn(5, 2)  # Changed to 2D array

        # NumPy backend
        np_LU, np_pivots = self.np_backend.lu_factor(A)
        np_result = self.np_backend.lu_solve((np_LU, np_pivots), b)

        # PyTorch backend
        torch_A = torch.from_numpy(A)
        torch_b = torch.from_numpy(b)
        torch_LU, torch_pivots = self.torch_backend.lu_factor(torch_A)
        torch_result = self.torch_backend.lu_solve((torch_LU, torch_pivots), torch_b)

        self.assert_close(np_result, torch_result)

    def test_norm(self):
        x = np.random.randn(5, 5)
        np_result = self.np_backend.norm(x)
        torch_result = self.torch_backend.norm(torch.from_numpy(x))
        self.assert_close(np_result, torch_result)

    def test_argsort(self):
        x = np.random.randn(5, 5)
        np_result = self.np_backend.argsort(x)
        torch_result = self.torch_backend.argsort(torch.from_numpy(x))
        self.assert_close(np_result, torch_result)

    def test_delete(self):
        x = np.random.randn(5, 5)
        np_result = self.np_backend.delete(x, 1, axis=0)
        torch_result = self.torch_backend.delete(torch.from_numpy(x), 1, axis=0)
        self.assert_close(np_result, torch_result)

    def test_block(self):
        a = np.random.randn(2, 2)
        b = np.random.randn(2, 2)
        c = np.random.randn(2, 2)
        d = np.random.randn(2, 2)
        np_result = self.np_backend.block([[a, b], [c, d]])
        torch_result = self.torch_backend.block([[torch.from_numpy(a), torch.from_numpy(b)],
                                                 [torch.from_numpy(c), torch.from_numpy(d)]])
        self.assert_close(np_result, torch_result)


if __name__ == '__main__':
    unittest.main()