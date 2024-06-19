import conv_mtx_idx as c
from GenericBackend import GenericBackend
import unittest

class Test_conv_mtx_idx(unittest.TestCase):
    def setUp(self) -> None:
        self.gbn = GenericBackend("numpy")
        self.gbt = GenericBackend("torch")
        
    def test_small_input(self):
        idx1 = [(0, 0), (1, 1)]
        idx2 = [(0, 1), (2, 0)]
        assert c.conv_mtx_idx_2d(idx1,idx2,self.gbt).tolist() == c.conv_mtx_idx_2d(idx1,idx2,self.gbn).tolist()

    def test_large_input(self):
        idx1 = [(i, j) for i in range(5) for j in range(3)]
        idx2 = [(i, j) for i in range(2) for j in range(4)]
        assert c.conv_mtx_idx_2d(idx1,idx2,self.gbt).tolist() == c.conv_mtx_idx_2d(idx1,idx2,self.gbn).tolist()

  #  def test_empty_input(self):
   #     idx1 = []
    #    idx2 = []
     #   assert ct.conv_mtx_idx_2d(idx1,idx2) == c.conv_mtx_idx_2d(idx1,idx2)
if __name__ == '__main__':
    unittest.main()    