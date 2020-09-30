import unittest
from qaoa import *
import numpy as np

class TestQAOA(unittest.TestCase):

    def test_kBits_MaxKCut(self):
        self.assertEqual(kBits_MaxKCut(2), 1)
        self.assertEqual(kBits_MaxKCut(3), 2)
        self.assertEqual(kBits_MaxKCut(4), 2)
        self.assertEqual(kBits_MaxKCut(5), 3)
        self.assertEqual(kBits_MaxKCut(6), 3)
        self.assertEqual(kBits_MaxKCut(7), 3)
        self.assertEqual(kBits_MaxKCut(8), 3)
        self.assertEqual(kBits_MaxKCut(9), 4)

    def test_binstringToLabels(self):

        num_V=1 # for the case of one vertex
        for k in [2,3,4,5,6,7,8,9]:
            num_qbits=kBits_MaxKCut(k)
            fo='{0:0'+str(num_qbits)+'b}'
            for i in range(k):
                i_binstring=fo.format(i)
                self.assertEqual(binstringToLabels_MaxKCut(k, num_V, i_binstring), str(i))

        num_V=2 # for the case of two vertices
        for k in [2,3,4,5,6,7,8,9]:
            num_qbits=kBits_MaxKCut(k)
            fo='{0:0'+str(num_qbits)+'b}'
            for i in range(k):
                i_binstring=fo.format(i)
                for j in range(k):
                    j_binstring=fo.format(j)
                    self.assertEqual(binstringToLabels_MaxKCut(k, num_V, i_binstring+j_binstring), str(i)+str(j))

        num_V=3 # for the case of two vertices
        for k in [2,3,4,5,6,7,8,9]:
            num_qbits=kBits_MaxKCut(k)
            fo='{0:0'+str(num_qbits)+'b}'
            for i in range(k):
                i_binstring=fo.format(i)
                for j in range(k):
                    j_binstring=fo.format(j)
                    for h in range(k):
                        h_binstring=fo.format(h)
                        self.assertEqual(binstringToLabels_MaxKCut(k, num_V, i_binstring+j_binstring+h_binstring), str(i)+str(j)+str(h))

if __name__ == '__main__':
    unittest.main()
