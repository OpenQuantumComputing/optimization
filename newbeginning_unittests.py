import unittest
from newbeginning import *
import numpy as np
from qiskit import *
import itertools

class QAOAst(QAOAbase):
    def createCircuit(self,string):
        qc = QuantumCircuit(self.R)
        for i in range(len(string)):
            if string[i] == "1":
                qc.x(i)
        return qc

backend = Aer.get_backend('statevector_simulator')

class TestQAOA(unittest.TestCase):

    def test_qaoa2qubit(self):
        for mu in [0,1,2.2]:

            FR = np.array([[1, 1],
                          [1, 1]])
            CR=np.array([1,10])

            solution={}
            solution['00'] = 0.0+2*mu
            solution['01'] = 10.0
            solution['10'] = 1.0
            solution['11'] = 11.0+2*mu


            qaoa = QAOAst(CR, FR)

            state_strings = np.array([''.join(i) for i in itertools.product('01', repeat=2)])

            for s in state_strings:

                # check cost function
                co, ex = qaoa.cost(s)
                self.assertAlmostEqual(co+mu*ex, solution[s])

                # check measurementstatistics function
                circuit = qaoa.createCircuit(s)
                job = execute(circuit, backend)
                res_dict = job.result().get_counts()
                statevector = job.result().results[0].data.statevector
                e,_,_ = qaoa.measurementStatistics(job, mu=mu, usestatevec=True)
                self.assertAlmostEqual(e, solution[s])
                e,_,_ = qaoa.measurementStatistics(job, mu=mu, usestatevec=False)
                self.assertAlmostEqual(e, solution[s])

    def test_qaoa2qubit(self):
        for mu in [0,1,2.2]:

            FR = np.array([[1, 1, 1],
                          [1, 1, 1]])
            CR=np.array([1,10,2])

            solution={}
            solution['000'] = 0.0+2*mu
            solution['001'] = 2.0
            solution['010'] = 10.0
            solution['011'] = 12.0+2*mu
            solution['100'] = 1.0
            solution['101'] = 3.0+2*mu
            solution['110'] = 11.0+2*mu
            solution['111'] = 13.0+8*mu


            qaoa = QAOAst(CR, FR)

            state_strings = np.array([''.join(i) for i in itertools.product('01', repeat=3)])

            for s in state_strings:
                #print(s, qaoa.cost(s, mu=mu))

                # check cost function
                co, ex = qaoa.cost(s)
                self.assertAlmostEqual(co+mu*ex, solution[s])

                # check measurementstatistics function
                circuit = qaoa.createCircuit(s)
                job = execute(circuit, backend)
                res_dict = job.result().get_counts()
                statevector = job.result().results[0].data.statevector

                e,_,_ = qaoa.measurementStatistics(job, mu=mu, usestatevec=True)
                self.assertAlmostEqual(e, solution[s])
                e,_,_ = qaoa.measurementStatistics(job, mu=mu, usestatevec=False)
                self.assertAlmostEqual(e, solution[s])


if __name__ == '__main__':
    unittest.main()
