from qiskit import *
from qiskit.tools.monitor import job_monitor

import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from qiskit.visualization import *
from scipy import optimize as opt
from qaoa import *
import os

import sys
sys.path.append('../')

from qiskit_utilities.utilities import *
from classical_maxkcut_solver import *


from matplotlib import rc
font = {'size' : 36}
rc('font', **font);
rc('text', usetex=True)
pl.rcParams["figure.figsize"] = 12,8
pl.rcParams["axes.titlesize"] = 24
pl.rcParams["axes.labelsize"] = 36
pl.rcParams["lines.linewidth"] = 3
pl.rcParams["lines.markersize"] = 10
pl.rcParams["xtick.labelsize"] = 36
pl.rcParams["ytick.labelsize"] = 36
#pl.style.use('bmh')

numV=10
G = nx.read_gml("../data/sample_graphs/w_ba_n"+str(numV)+"_k4_0.gml")

#pos = nx.spring_layout(G)
#nx.draw_networkx(G,apos=pos)

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')

beta_n = 12
gamma_n = 24
beta_max = np.pi/2
gamma_max = np.pi
optmethod='Nelder-Mead'
circuit_version=2
shots=1024*2*2*2
rerun=False

maxdepth=3

depths=range(1,maxdepth+1)

outstr=""
outstrbest=""

max_val = np.array([8.657714089848158, 10.87975400338161, 11.059417685176726, 11.059417685176726, 11.059417685176726, 11.059417685176726, 11.059417685176726])

for k_cuts in [2,3,4,5,6,7,8]:
    if k_cuts<=4:
        backend = Aer.get_backend('qasm_simulator')
    else:
        backend = provider.get_backend('ibmq_qasm_simulator')

    #max_val, label = find_max_cut_brute_force(G, k_cuts)
    #classical_maxkcut_solver(G, k_cuts)

    outstr+=str(k_cuts)
    outstrbest+=str(k_cuts)
    print(" k=", k_cuts)
    name="BA10w"+str(gamma_n)+"x"+str(beta_n)+"_v"+str(circuit_version)+"_k"+str(k_cuts)
    Elandscape, gammabetas, E, best =  runQAOA(G, k_cuts, backend, gamma_n, beta_n, gamma_max, beta_max, optmethod=optmethod, circuit_version=circuit_version, shots=shots, name=name, rerun=rerun)

    shiftg=gamma_max/(2*gamma_n)
    shiftb=beta_max/(2*beta_n)

    pl.figure(figsize=(20,10));
    pl.clf()
    pl.imshow(Elandscape,interpolation='spline36',origin='lower'
                ,extent=[-shiftg,gamma_max+shiftg,-shiftb,beta_max+shiftb], aspect=1)
    pl.xticks([0,gamma_max/2, gamma_max], ['0', r'$\pi$', r'$2\pi$'])
    pl.yticks([0,beta_max], ['0', r'$\pi/2$'])
    pl.xlabel('$\gamma$',loc='left')
    pl.ylabel(r'$\beta$')
    pl.colorbar(shrink=0.75, pad=0.05, orientation="horizontal")


    #pl.plot(gammabetas['x0_d1'][0], gammabetas['x0_d1'][1],'xw')
    pl.plot(gammabetas['xL_d1'][0], gammabetas['xL_d1'][1],'or')

    pl.tight_layout()
    pl.savefig("pics/E_"+name+".png")
    pl.close()
    print("saved", "pics/E_",name,".png")

    #num_shots, av_max_cost, best_cost, distribution = getStatistics(G, k_cuts, backend, gammabetas, circuit_version=circuit_version, shots=shots, name=name+"_best")

    for depth in depths:
        outstr+=" & "+"{:.2f}".format(E[str(depth)]/max_val[k_cuts-2])
    outstr+="\\\\ \n"
    print(outstr)

    for depth in depths:
        outstrbest+=" & "+"{:.2f}".format(best[str(depth)]/max_val[k_cuts-2])
    outstrbest+="\\\\ \n"
    print(outstrbest)

    #pl.figure(figsize=(20,10))
    #pl.clf()
    #for depth in depths:
    #    pl.semilogx(num_shots[63:], np.array(av_max_cost[str(depth)][63:])/max_val[k_cuts-2],'x-', base=2, label='d='+str(depth), linewidth=2,markersize=14)
    #pl.ylabel('r')
    #pl.xlabel('num shots')
    #pl.legend()
    #pl.grid(True, which="both")
    #pl.tight_layout()
    #pl.savefig("pics/"+name+"_avE.png")
    #pl.close()

    #pl.figure(figsize=(20,10))
    #pl.clf()
    #for depth in depths:
    #    pl.plot(num_shots[:36], np.array(av_max_cost[str(depth)][:36])/max_val[k_cuts-2],'x-', label='d='+str(depth), linewidth=2,markersize=14)
    #pl.legend()
    #pl.ylabel('r')
    #pl.xlabel('num shots')
    #pl.legend()
    #pl.grid(True, which="both")
    #pl.tight_layout()
    #pl.savefig("pics/"+name+"_bestE"+str(depth)+".png")
    #pl.close()

    pl.figure(figsize=(20,10))
    pl.clf()
    for d in depths:
        if d==1:
            col='r'
        elif d==2:
            col='g'
        else:
            col='b'
        sty=''
        if not d==1:
            sty=':'
        pl.plot(np.arange(1,d+1),gammabetas['x0_d'+str(d)][::2],'x'+col+sty,label='depth='+str(d))
        sty=''
        if not d==1:
            sty='-'
        pl.plot(np.arange(1,d+1),gammabetas['xL_d'+str(d)][::2],'o'+col+sty,label='depth='+str(d))
    pl.xlabel('depth')
    pl.ylabel(r'$\gamma$')
    pl.legend()
    pl.tight_layout()
    pl.savefig("pics/"+name+"_gamma.png")
    pl.close()

    pl.figure(figsize=(20,10))
    pl.clf()
    for d in depths:
        if d==1:
            col='r'
        elif d==2:
            col='g'
        else:
            col='b'
        sty=''
        if not d==1:
            sty=':'
        pl.plot(np.arange(1,d+1),gammabetas['x0_d'+str(d)][1::2],'x'+col+sty,label='depth='+str(d))
        sty=''
        if not d==1:
            sty='-'
        pl.plot(np.arange(1,d+1),gammabetas['xL_d'+str(d)][1::2],'o'+col+sty,label='depth='+str(d))
    pl.legend()
    pl.xlabel('depth')
    pl.ylabel(r'$\beta$')
    pl.legend()
    pl.tight_layout()
    pl.savefig("pics/"+name+"_beta.png")
    pl.close()



