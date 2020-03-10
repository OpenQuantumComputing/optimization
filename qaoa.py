from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

def createCircuit_MaxCut(x,G,depth,version=1,usebarrier=False):
    num_V = G.number_of_nodes()
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
            w = G[edge[0]][edge[1]]['weight']
            wg = w*gamma
            if version==1:
                circ.cx(q[i],q[j])
                circ.rz(wg,q[j])
                circ.cx(q[i],q[j])
            else:
                circ.cu1(-2*wg, i, j)
                circ.u1(wg, i)
                circ.u1(wg, j)
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
        w = G[edge[0]][edge[1]]['weight']
        C = C + w/2*(1-(2*x[i]-1)*(2*x[j]-1))
    return C

def listSortedCosts_MaxCut(G):
    costs={}
    maximum=0
    solutions=[]
    num_V = G.number_of_nodes()
    for i in range(2**num_V):
        binstring="{0:b}".format(i).zfill(num_V)
        y=[int(i) for i in binstring]
        costs[binstring]=cost_MaxCut(y,G)
    sortedcosts={k: v for k, v in sorted(costs.items(), key=lambda item: item[1])}
    return sortedcosts

