###for testing only
import sys
sys.path.append("C:/Users/w-a-c/Desktop/inkstone")
###----------------

import params as p
from GenericBackend import GenericBackend
import unittest   # The test framework

class Test_params(unittest.TestCase):
    def setUp(self) -> None:
        self.gbn = GenericBackend("numpy")
        self.gbt = GenericBackend("torch")
        self.a = p.Params(gb=self.gbn)
        self.a.lattice = 1
        self.a.num_g = 30
        self.a.frequency = 0.4
        
        self.b = p.Params(gb=self.gbt)
        self.b.lattice = 1
        self.b.num_g = 30
        self.b.frequency = 0.4
    
    
    def test_latt_vec(self):# use .__dict__ to find all attributes
        self.a.latt_vec = 3.14
        self.b.latt_vec = 3.14
        for i in range(len(self.a.gs)):
            assert self.a.gs[i][0] == self.b.gs[i][0].item()
            assert self.a.gs[i][1] == self.b.gs[i][1].item() 
        
        assert self.a.idx_g == self.b.idx_g
        
        assert self.a.idx_conv_mtx.tolist() == self.b.idx_conv_mtx.tolist()
        assert self.a.idx_g_ep_mu_used == self.b.idx_g_ep_mu_used 
        assert self.a.phif.tolist() == self.b.phif.tolist() 
        assert self.a._omega == self.b._omega
        

if __name__ == '__main__':
    unittest.main()