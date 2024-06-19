import unittest
import numpy as np
from GenericBackend import GenericBackend
import ft.ft_1d_sq as f



class TestFt1dSq(unittest.TestCase):
    def setUp(self) -> None:
        self.gbn = GenericBackend("numpy")
        self.gbt = GenericBackend("torch")
        
    def test_center_zero_width_one(self):
        ks = np.linspace(-10, 10, 1000).tolist()
        s = f.ft_1d_sq(1, ks, gb=self.gbn)
        st = f.ft_1d_sq(1,ks, gb=self.gbt)
        
        assert s == st

    def test_center_zero_width_two(self):
        ks = np.linspace(-10, 10, 1000)
        s = f.ft_1d_sq(1, ks,gb=self.gbn)
        st = f.ft_1d_sq(1,ks,gb=self.gbt)
        
        assert s == st


    def test_center_one_width_one(self):
        ks = np.linspace(-10, 10, 1000)
        s = f.ft_1d_sq(1, ks,gb=self.gbn)
        st = f.ft_1d_sq(1,ks,gb=self.gbt)
        
        assert s == st

    def test_empty_ks(self):
        ks = []
        s = f.ft_1d_sq(1, ks,gb=self.gbn)
        st = f.ft_1d_sq(1,ks,gb=self.gbt)
        
        assert s == st


    def test_negative_width(self):
        ks = np.linspace(-10, 10, 1000)
        with self.assertRaises(ValueError):
            s = f.ft_1d_sq(-1, ks,gb=self.gbn)
            st = f.ft_1d_sq(-1,ks,gb=self.gbt)



if __name__ == '__main__':
    unittest.main()