import unittest
import ft.ft_2d_disk as f
from GenericBackend import GenericBackend

class TestFt2dDisk(unittest.TestCase):
    def setUp(self) -> None:
        self.gbn = GenericBackend("numpy")
        self.gbt = GenericBackend("torch")
    
    def test_origin_center_unit_radius(self):
        r = 1.0
        ks = [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
        assert f.ft_2d_disk(r,ks,gb=self.gbn) == f.ft_2d_disk(r,ks,gb=self.gbt)

    def test_origin_center_radius_two(self):
        r = 2.0
        ks = [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
        assert f.ft_2d_disk(r,ks,gb=self.gbn) == f.ft_2d_disk(r,ks,gb=self.gbt)


    def test_nonzero_center(self):
        r = 1.0
        center = (0.5, 0.5)
        ks = [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
        assert f.ft_2d_disk(r,ks,center,gb=self.gbn) == f.ft_2d_disk(r,ks,center,gb=self.gbt)


    def test_empty_ks(self):
        r = 1.0
        ks = []
        assert f.ft_2d_disk(r,ks,gb=self.gbn) == f.ft_2d_disk(r,ks,gb=self.gbt)


if __name__ == '__main__':
    unittest.main()