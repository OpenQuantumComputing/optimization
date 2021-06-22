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

    def test_is_Soluton(self):

        """
        Test-matrix from 
        https://github.com/marikasvenssonjeppesen/Extracted-datainstances-for-Tailassignment/blob/master/data_instances/instance_8_0no_feas_sol_1_small_costs.simple

    
        """

        ind = [ [7,1,3,5,6,8,10,13,14,16,18,20,22],
                [1,0,2,4,7,9,11,12,15,17,19,21,23],
                [5,1,3,5,7,9,11,13,15,16,19,21,23],
                [2,1,3,5,6,8,11,13,15,17,19,20,22],
                [3,1,3,5,7,9,11,13,15,16,18,20,22],
                [8,0,2,4,6,8,10,12,14,16,18,21,23],
                [6,1,3,5,6,8,11,12,14,16,18,20,23],
                [4,1,3,5,6,8,11,13,15,17,19,21,22]]

        FR = np.zeros((24,8))

        for i,l in enumerate(ind):
            FR[l[1:],l[0] - 1] = 1

        # Only one solution
        self.assertEqual(is_Solution('10000010'[::-1],FR), True)

        # Not solutions
        self.assertEqual(is_Solution('10000100'[::-1],FR), False)
        self.assertEqual(is_Solution('01000010'[::-1],FR), False)
        self.assertEqual(is_Solution('00010010'[::-1],FR), False)
        self.assertEqual(is_Solution('00101000'[::-1],FR), False)
        self.assertEqual(is_Solution('10010000'[::-1],FR), False)
        self.assertEqual(is_Solution('10001000'[::-1],FR), False)
        self.assertEqual(is_Solution('10010000'[::-1],FR), False)

if __name__ == '__main__':
    unittest.main()
