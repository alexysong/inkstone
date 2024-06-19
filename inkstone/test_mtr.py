import mtr as m

import unittest

class Test_mtr(unittest.TestCase):
    def test1(self):
        a = m.Mtr(3.2,4.7,name="foo")
        b = m.Mtr(3.2,4.7,name="foo")
        
        
        print(a.__dict__)
        print(b.__dict__)
        pass

  #  def test_empty_input(self):
   #     idx1 = []
    #    idx2 = []
     #   assert m.max_idx_diff(idx1,idx2) == mt.max_idx_diff(idx1,idx2)




if __name__ == '__main__':
    unittest.main()    