import ft.ft_2d_cnst as f
from GenericBackend import GenericBackend
import unittest


class TestFt2dCnst(unittest.TestCase):
    def setUp(self) -> None:
        self.gbn = GenericBackend("numpy")
        self.gbt = GenericBackend("torch")
    
    def test_origin_only(self):
        ks = [(0, 0)]
        assert f.ft_2d_cnst(ks,self.gbn) == f.ft_2d_cnst(ks,self.gbt)

    def test_nonzero_values(self):
        ks = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        assert f.ft_2d_cnst(ks,self.gbn) == f.ft_2d_cnst(ks,self.gbt)


    def test_mixed_values(self):
        ks = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
        assert f.ft_2d_cnst(ks,self.gbn) == f.ft_2d_cnst(ks,self.gbt)

    def test_empty_ks(self):
        ks = []
        assert f.ft_2d_cnst(ks,self.gbn) == f.ft_2d_cnst(ks,self.gbt)

if __name__ == '__main__':
    unittest.main()