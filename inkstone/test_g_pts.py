import g_pts as g
from GenericBackend import GenericBackend
import unittest

class TestGPts(unittest.TestCase):
    def setUp(self) -> None:
        self.gbn = GenericBackend("numpy")
        self.gbt = GenericBackend("torch")
        
    def test_rectangular_lattice(self):
        num_g = 10
        b1 = (1.0, 0.0)
        b2 = (0.0, 1.0)

        k, i = g.g_pts(num_g, b1,b2,self.gbt)
        kk, ii = g.g_pts(num_g,b1,b2,self.gbn)
        assert len(k) == len(kk)
        assert len(i) == len(ii)
        
        for j in range(len(k)):
            assert k[j][0] == kk[j][0]
            assert i[j][0] == ii[j][0]
            
    def test_non_orthogonal_lattice(self):
        num_g = 10
        b1 = (1.0, 1.0)
        b2 = (1.0, -1.0)

        k, i = g.g_pts(num_g, b1,b2,self.gbn)
        kk, ii = g.g_pts(num_g,b1,b2,self.gbt)
        assert len(k) == len(kk)
        assert len(i) == len(ii)
        
        for j in range(len(k)):
            assert k[j][0] == kk[j][0]
            assert i[j][0] == ii[j][0]

    def test_zero_num_g(self):
        num_g = 0
        b1 = (1.0, 0.0)
        b2 = (0.0, 1.0)
        
        k, i = g.g_pts(num_g, b1,b2,self.gbt)
        kk, ii = g.g_pts(num_g,b1,b2,self.gbn)
        assert len(k) == len(kk)
        assert len(i) == len(ii)       
        
        for j in range(len(k)):
            assert k[j][0] == kk[j][0]
            assert i[j][0] == ii[j][0]
        


if __name__ == '__main__':
    unittest.main()
    

