import recipro as r
from GenericBackend import GenericBackend
import unittest
#numpy to torch.float64 in ppt
class Test_recipro(unittest.TestCase):
    def setUp(self) -> None:
        self.gbn = GenericBackend("numpy")
        self.gbt = GenericBackend("torch")
        
    def test1(self):
        p,q = r.recipro((3.0,1.0),(3.0,4.0),self.gbn)
        a,b = r.recipro((3.0,1.0),(3.0,4.0),self.gbt)
        assert (p == [x.item() for x in a]).all()
        assert (q == [x.item() for x in b]).all()
        
    def test2(self):
        p,q = r.recipro((1.7,3.4),(6.5,2.8),self.gbn)
        a,b = r.recipro((1.7,3.4),(6.5,2.8),self.gbt)
        assert (p == tuple([x.item() for x in a])).all()
        assert (q == tuple([x.item() for x in b])).all()
    
    def test3(self):
        p, q = r.recipro((0.0, 0.0), (2.1, -1.3),self.gbn)
        a, b = r.recipro((0.0, 0.0), (2.1, -1.3),self.gbt)
        assert p == (float('inf'), float('inf'))
        assert q == tuple([x.item() for x in b])
    
    def test4(self):
        p, q = r.recipro((4.2, 1.6), (0.0, 0.0),self.gbn)
        a, b = r.recipro((4.2, 1.6), (0.0, 0.0),self.gbt)
        assert p == tuple([x.item() for x in a])
        assert q == (float('inf'), float('inf'))

    def test5(self):
        with self.assertRaises(Exception) as cm:
            r.recipro((0.0, 0.0), (0.0, 0.0),self.gbn)
        self.assertEqual(str(cm.exception), "The two lattice vectors can't be both zero vectors.")

        with self.assertRaises(Exception) as cm:
            r.recipro((0.0, 0.0), (0.0, 0.0),self.gbt)
        self.assertEqual(str(cm.exception), "The two lattice vectors can't be both zero vectors.")

    def test6(self):
        p, q = r.recipro((1e6, 1e-6), (1e-6, 1e6),self.gbn)
        a, b = r.recipro((1e6, 1e-6), (1e-6, 1e6),self.gbt)
        assert (p == [x.item() for x in a]).all()
        assert (q == [x.item() for x in b]).all()

    def test7(self):
        p, q = r.recipro((-2.5, 1.8), (-3.7, -0.9),self.gbn)
        a, b = r.recipro((-2.5, 1.8), (-3.7, -0.9),self.gbt)
        assert (p == tuple([x.item() for x in a])).all()
        assert (q == tuple([x.item() for x in b])).all()
if __name__ == '__main__':
    unittest.main()    