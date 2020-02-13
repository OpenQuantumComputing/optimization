from qiskit import *

def createCircuit_MaxCut(x,G,depth,usebarrier=False):
    V = list(G.nodes)
    num_V = len(V)
    q = QuantumRegister(num_V)
    c = ClassicalRegister(num_V)
    circ = QuantumCircuit(q,c)
    circ.h(range(num_V))
    if usebarrier:
        circ.barrier()
    for d in range(depth):
        gamma=x[2*d]
        beta=x[2*d+1]
        for edge in G.edges():
            i=int(edge[0])
            j=int(edge[1])
            w = G[i][j]['weight']
            circ.cx(q[i],q[j])
            circ.rz(w*gamma,q[j])
            circ.cx(q[i],q[j])
        if usebarrier:
            circ.barrier()
        circ.rx(2*beta,range(num_V))
        if usebarrier:
            circ.barrier()
    circ.measure(q,c)
    return circ

def cost_MaxCut(x,G):
    C=0
    for edge in G.edges():
        i = int(edge[0])
        j = int(edge[1])
        w = G[i][j]['weight']
        C = C + w/2*(1-(2*x[i]-1)*(2*x[j]-1))
    return C

def listcosts_MaxCut(G):
    costs={}
    maximum=0
    solutions=[]
    V = list(G.nodes)
    num_V = len(V)
    for i in range(2**num_V):
        binstring="{0:b}".format(i).zfill(num_V)
        y=[int(i) for i in binstring]
        costs[binstring]=cost_MaxCut(y,G)
        maximum = max(maximum,costs[binstring])
    for key in costs:
        if costs[key]==maximum:
            solutions.append(key)
    return costs, maximum, solutions
