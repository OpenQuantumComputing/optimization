import unittest
from exactcover import *
import numpy as np
from qiskit import *

class TestExactCover(unittest.TestCase):

    def test_(self):
        FR = np.zeros((2,4))
        FR[0,1]=1
        FR[1,2]=1
        FR[0,3]=1
        FR[1,3]=1

        ### inverse order of the bit strings, because qiskit order is $q_n q_{n-1} .... q_0$
        ## solutions
        self.assertEqual(cost_exactCover('0110'[::-1],FR)==0, True)
        self.assertEqual(cost_exactCover('0001'[::-1],FR)==0, True)
        self.assertEqual(cost_exactCover('1001'[::-1],FR)==0, True)
        self.assertEqual(cost_exactCover('1110'[::-1],FR)==0, True)

        ## not solutions
        self.assertEqual(cost_exactCover('0000'[::-1],FR)<0, True)
        self.assertEqual(cost_exactCover('0010'[::-1],FR)<0, True)
        self.assertEqual(cost_exactCover('0011'[::-1],FR)<0, True)
        self.assertEqual(cost_exactCover('0100'[::-1],FR)<0, True)
        self.assertEqual(cost_exactCover('0101'[::-1],FR)<0, True)
        self.assertEqual(cost_exactCover('0111'[::-1],FR)<0, True)
        self.assertEqual(cost_exactCover('1000'[::-1],FR)<0, True)
        self.assertEqual(cost_exactCover('1010'[::-1],FR)<0, True)
        self.assertEqual(cost_exactCover('1011'[::-1],FR)<0, True)
        self.assertEqual(cost_exactCover('1100'[::-1],FR)<0, True)
        self.assertEqual(cost_exactCover('1101'[::-1],FR)<0, True)
        self.assertEqual(cost_exactCover('1111'[::-1],FR)<0, True)

if __name__ == '__main__':
    unittest.main()
