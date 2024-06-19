import max_idx_diff as m
from GenericBackend import GenericBackend
import unittest

class Test_max_idx_diff(unittest.TestCase):
    def setUp(self) -> None:
        self.gbn = GenericBackend("numpy")
        self.gbt = GenericBackend("torch")
        
    def test_small_input(self):
        idx1 = [(0, 0), (1, 1)]
        a,b = m.max_idx_diff(idx1,self.gbn)
        c,d = m.max_idx_diff(idx1,self.gbt)
        assert a == c.item()
        assert b == d.item()
        

    def test_large_input(self):
        idx1 = [(i, j) for i in range(5) for j in range(3)]
        a,b = m.max_idx_diff(idx1,self.gbn)
        c,d = m.max_idx_diff(idx1,self.gbt)
        assert a == c.item()
        assert b == d.item()

  #  def test_empty_input(self):
   #     idx1 = []
    #    idx2 = []
     #   assert m.max_idx_diff(idx1,idx2) == m.max_idx_diff(idx1,idx2)
if __name__ == '__main__':
    unittest.main()    