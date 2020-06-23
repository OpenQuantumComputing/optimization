from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import numpy as np

def createCircuit_MaxCut(x,G,depth,version=1, applyX=[], usebarrier=False):
    num_V = G.number_of_nodes()
    q = QuantumRegister(num_V)
    c = ClassicalRegister(num_V)
    circ = QuantumCircuit(q,c)
    if len(applyX)==0:
        circ.h(range(num_V))
    else:
        if np.where(np.array(applyX)==1)[0].size>0:
            circ.x(np.where(np.array(applyX)==1)[0])
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

def enumerate(G):
    if (len(G) > 30):
        raise Exception("Too many solutions to enumerate.")

    maxcut = []
    maxcut_value = 0
    N = len(G)
    for i in range(2**N - 1):
        x_bin = format(i, 'b').zfill(N)
        x = [int(j) for j in x_bin]
        c = 0
        for u,v in G.edges():
            c += G[u][v]['weight']/2*(1-(2*x[int(u)]-1)*(2*x[int(v)]-1))

        if (c > maxcut_value):
            maxcut = x
            maxcut_value = c

    return maxcut_value, maxcut

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

def costsHist_MaxCut(G):
    num_V = G.number_of_nodes()
    costs=np.ones(2**num_V)
    for i in range(2**num_V):
        if i%1024*2*2*2==0:
            print(i/2**num_V*100, "%", end='\r')
        binstring="{0:b}".format(i).zfill(num_V)
        y=[int(i) for i in binstring]
        costs[i]=cost_MaxCut(y,G)
    print("100%")
    return costs

def bins_comp_basis(data, G):
    max_solutions=[]
    num_V = G.number_of_nodes()
    bins_states = np.zeros(2**num_V)
    num_shots=0
    num_solutions=0
    max_cost=0
    average_cost=0
    for item, binary_rep in enumerate(data):
        integer_rep=int(str(binary_rep), 2)
        counts=data[str(binary_rep)]
        bins_states[integer_rep] += counts
        num_shots+=counts
        num_solutions+=1
        y=[int(i) for i in str(binary_rep)]
        lc = cost_MaxCut(y,G)
        if lc==max_cost:
            max_solutions.append(y)
        elif lc>max_cost:
            max_solutions=[]
            max_solutions.append(y)
            max_cost=lc
        average_cost+=lc*counts
    return bins_states, max_cost, average_cost/num_shots, max_solutions

def expectationValue_MaxCut(data,G):
    E=[]
    V = list(G.nodes)
    num_qubits = len(V)
    for item in range(0,len(data)):
        shots = data[item].shots
        counts = data[item].data.counts
        E.append(0)
        for key in list(counts.__dict__.keys()):
            c=getattr(counts, key)#number of counts
            binstring="{0:b}".format(int(key,0)).zfill(num_qubits)
            y=[int(i) for i in binstring]
            E[item] += cost_MaxCut(y,G)*c/shots
    return E
