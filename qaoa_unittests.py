import unittest
from qaoa import *
import numpy as np
from qiskit import *

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

    def test_getcolor(self):

        self.assertEqual(getcolor("00"), -1)

        self.assertEqual(getcolor("10"), 0)
        self.assertEqual(getcolor("01"), 1)

        self.assertEqual(getcolor("100"), 0)
        self.assertEqual(getcolor("010"), 1)
        self.assertEqual(getcolor("001"), 2)

    def test_binstringToLabels_MaxKCut_onehot(self):

        num_V=1
        k=2
        self.assertEqual(binstringToLabels_MaxKCut_onehot("10", num_V, k), "0")
        self.assertEqual(binstringToLabels_MaxKCut_onehot("01", num_V, k), "1")

        k=3
        self.assertEqual(binstringToLabels_MaxKCut_onehot("100", num_V, k), "0")
        self.assertEqual(binstringToLabels_MaxKCut_onehot("010", num_V, k), "1")
        self.assertEqual(binstringToLabels_MaxKCut_onehot("001", num_V, k), "2")

        num_V=2
        k=2
        self.assertEqual(binstringToLabels_MaxKCut_onehot("1001", num_V, k), "01")
        self.assertEqual(binstringToLabels_MaxKCut_onehot("0101", num_V, k), "11")
        self.assertEqual(binstringToLabels_MaxKCut_onehot("0110", num_V, k), "10")
        self.assertEqual(binstringToLabels_MaxKCut_onehot("1010", num_V, k), "00")

        k=3
        self.assertEqual(binstringToLabels_MaxKCut_onehot("100100", num_V, k), "00")
        self.assertEqual(binstringToLabels_MaxKCut_onehot("010100", num_V, k), "10")
        self.assertEqual(binstringToLabels_MaxKCut_onehot("001100", num_V, k), "20")
        self.assertEqual(binstringToLabels_MaxKCut_onehot("100010", num_V, k), "01")
        self.assertEqual(binstringToLabels_MaxKCut_onehot("010010", num_V, k), "11")
        self.assertEqual(binstringToLabels_MaxKCut_onehot("001010", num_V, k), "21")
        self.assertEqual(binstringToLabels_MaxKCut_onehot("100001", num_V, k), "02")
        self.assertEqual(binstringToLabels_MaxKCut_onehot("010001", num_V, k), "12")
        self.assertEqual(binstringToLabels_MaxKCut_onehot("001001", num_V, k), "22")


    def test_validcoloring_onehot(self):

        self.assertEqual(validstring_onehot("00",1), False)
        self.assertEqual(validstring_onehot("01",1), True)
        self.assertEqual(validstring_onehot("10",1), True)
        self.assertEqual(validstring_onehot("11",1), False)

        self.assertEqual(validstring_onehot("0000",2), False)
        self.assertEqual(validstring_onehot("0001",2), False)
        self.assertEqual(validstring_onehot("0010",2), False)
        self.assertEqual(validstring_onehot("0011",2), False)
        self.assertEqual(validstring_onehot("0100",2), False)
        self.assertEqual(validstring_onehot("0101",2), True)
        self.assertEqual(validstring_onehot("0110",2), True)
        self.assertEqual(validstring_onehot("0111",2), False)
        self.assertEqual(validstring_onehot("1000",2), False)
        self.assertEqual(validstring_onehot("1001",2), True)
        self.assertEqual(validstring_onehot("1010",2), True)
        self.assertEqual(validstring_onehot("1011",2), False)
        self.assertEqual(validstring_onehot("1100",2), False)
        self.assertEqual(validstring_onehot("1101",2), False)
        self.assertEqual(validstring_onehot("1110",2), False)
        self.assertEqual(validstring_onehot("1111",2), False)

        self.assertEqual(validstring_onehot("001100",3), False)
        self.assertEqual(validstring_onehot("001100",2), True)

    def test_validcoloring_onehot(self):

        backend_sv = Aer.get_backend('statevector_simulator')

        for k_cuts in [2,3,4,5,6,7,8]:
            q = QuantumRegister(k_cuts)
            c = ClassicalRegister(k_cuts)
            circ = QuantumCircuit(q, c)
            Wn(circ, [i for i in range(k_cuts)])
            circ.draw()
            job = execute(circ, backend_sv)
            result = job.result()
            outputstate = result.get_statevector(circ)
            sv1=qiskit.quantum_info.Statevector(outputstate)

            W=np.zeros(2**k_cuts)
            for i in range(k_cuts):
                bs='0b'
                for j in range(k_cuts):
                    if i == j:
                        bs+="1"
                    else:
                        bs+="0"
                ind=int(bs, 2)
                W[ind]=1
            W = W/np.sqrt(k_cuts)
            sv2=qiskit.quantum_info.Statevector(W)

            self.assertEqual(sv1.equiv(sv2), True)

if __name__ == '__main__':
    unittest.main()
