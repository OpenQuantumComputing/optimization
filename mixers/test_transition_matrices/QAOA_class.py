from openquantumcomputing.QAOABase import QAOABase
from qiskit import *

class QAOA(QAOABase):
    def cost(self, string, params):
        cost_array = params.get("cost_array")
        return cost_array[int(string, 2)]

    def createCircuit(self, angles, depth, params={}):
        graph = params.get("graph")
        mixer_array = params.get("mixer_array")
        num_V = graph.number_of_nodes()

        q = QuantumRegister(num_V)
        c = ClassicalRegister(num_V)
        circ = QuantumCircuit(q, c)

        circ.h(q)

        for d in range(depth):
            gamma = angles[2 * d]
            beta = angles[2 * d + 1]
            # cost Hamiltonian
            for i, j, data in graph.edges.data():
                i, j = int(i), int(j)
                circ.cx(q[i], q[j])
                circ.rz(data["weight"] * gamma, q[j])
                circ.cx(q[i], q[j])

            # mixer Hamiltonian
            mixer_array.add_gates(circ, beta)

        circ.measure(q, c)
        return circ
