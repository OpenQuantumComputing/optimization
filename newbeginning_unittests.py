import unittest
from newbeginning import *
import numpy as np
from qiskit import *

class QAOA00(QAOAbase):
    def createCircuit(self):
        F, R  = np.shape(self.FR)
        qc = QuantumCircuit(R)
        return qc

class QAOA10(QAOAbase):
    def createCircuit(self):
        F, R  = np.shape(self.FR)
        qc = QuantumCircuit(R)
        qc.x(0)
        return qc
class QAOA01(QAOAbase):
    def createCircuit(self):
        F, R  = np.shape(self.FR)
        qc = QuantumCircuit(R)
        qc.x(1)
        return qc
class QAOA11(QAOAbase):
    def createCircuit(self):
        F, R  = np.shape(self.FR)
        qc = QuantumCircuit(R)
        qc.x(0)
        qc.x(1)
        return qc

Aer.backends()
backend = Aer.get_backend('statevector_simulator')

class TestQAOA(unittest.TestCase):

    def test_qaoa(self):

        FR = np.array([[1, 1],
                      [1, 1]])
        CR=np.array([1,10])
        # Normalize weights
        CR = CR/np.max(CR)

        qaoa = QAOA00(CR, FR)
        print("cost(00)=", qaoa.cost('00', mu=0))
        print("cost(01)=", qaoa.cost('01', mu=0))
        print("cost(10)=", qaoa.cost('10', mu=0))
        print("cost(11)=", qaoa.cost('11', mu=0))

        qaoa = QAOA00(CR, FR)
        circuit = qaoa.createCircuit()
        job = execute(circuit, backend)
        res_dict = job.result().get_counts()
        print("dict=",res_dict)
        statevector = job.result().results[0].data.statevector
        print(statevector)

        print(qaoa.measurementStatistics(job, mu=0, usestatevec=True))
        print(qaoa.measurementStatistics(job, mu=0, usestatevec=False))

        qaoa = QAOA01(CR, FR)
        circuit = qaoa.createCircuit()
        job = execute(circuit, backend)
        res_dict = job.result().get_counts()
        print("dict=",res_dict)
        statevector = job.result().results[0].data.statevector
        print(statevector)

        print(qaoa.measurementStatistics(job, mu=0, usestatevec=True))
        print(qaoa.measurementStatistics(job, mu=0, usestatevec=False))
        


if __name__ == '__main__':
    unittest.main()
