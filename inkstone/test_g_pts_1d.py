import g_pts_1d as g
from GenericBackend import GenericBackend
import unittest

class TestGPts1D(unittest.TestCase):
    def setUp(self) -> None:
        self.gbn = GenericBackend("numpy")
        self.gbt = GenericBackend("torch")
    
    def test_even_num_g(self):
        num_g = 10
        b = (1.0, 0.0)
      #  expected_k_pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (-4.0, 0.0), (-3.0, 0.0), (-2.0, 0.0), (-1.0, 0.0)]
      #  expected_idx = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

        k, i = g.g_pts_1d(num_g, b, self.gbn)
        kk, ii = g.g_pts_1d(num_g,b,self.gbt)
        assert kk == k
        assert ii == i

    def test_odd_num_g(self):
        num_g = 9
        b = (0.5, 0.5)
     #   expected_k_pts = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (1.5, 1.5), (2.0, 2.0), (-2.0, -2.0), (-1.5, -1.5), (-1.0, -1.0), (-0.5, -0.5)]
      #  expected_idx = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

        k, i = g.g_pts_1d(num_g, b,self.gbn)
        kk, ii = g.g_pts_1d(num_g,b,self.gbt)
        assert kk == k
        assert ii == i

    def test_zero_num_g(self):
        num_g = 0
        b = (1.0, 1.0)
    #    expected_k_pts = []
     #   expected_idx = []

        k, i = g.g_pts_1d(num_g, b,self.gbn)
        kk, ii = g.g_pts_1d(num_g,b,self.gbt)
        assert kk == k
        assert ii == i

    def test_negative_num_g(self):
        num_g = -5
        b = (0.5, 0.5)
    #    expected_k_pts = []
    #    expected_idx = []

        k, i = g.g_pts_1d(num_g, b,self.gbn)
        kk, ii = g.g_pts_1d(num_g,b,self.gbt)
        assert kk == k
        assert ii == i

if __name__ == '__main__':
    unittest.main()