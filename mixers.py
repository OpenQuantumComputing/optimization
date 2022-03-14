import numpy as np
from sympy.physics.quantum import TensorProduct
from sympy.physics.paulialgebra import Pauli, evaluate_pauli_product
from binsymbols import *
from sympy import *
import itertools
import math

X=Pauli(1)
Y=Pauli(2)
Z=Pauli(3)
I=1

def all_states(n):
    return [''.join(i) for i in itertools.product('01', repeat= n)]

def hamming(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def get_T(n, mode, d=1,i=None,j=None, shift=1, oddeven="both"):
    if mode=="leftright":
        T=np.zeros((n,n))
        if isinstance(i,list):
            if not len(i)==len(j):
                raise ValueError('mode "'+mode+'": lenght of index lists must be equal')
            for ind in range(len(i)):
                T[i[ind],j[ind]]=1
                T[j[ind],i[ind]]=1
        else:
            T[i,j]=1
            T[j,i]=1
    elif mode=="full":
        T=np.ones((n,n))
        for i in range(n):
            T[i,i]=0
    elif mode=="nearest_int":
        T=np.zeros((n,n))
        for i in range(n-shift):
            if oddeven=="even" and i%2==0:
                continue
            elif oddeven=="odd" and i%2==1:
                continue
            T[i,i+shift]=1
            T[i+shift,i]=1
    elif mode=="nearest_int_cyclic":
        T=np.zeros((n,n))
        for i in range(n-1):
            if oddeven=="even" and i%2==0:
                continue
            elif oddeven=="odd" and i%2==1:
                continue
            T[i,i+1]=1
            T[i+1,i]=1
        if not oddeven=="even":
            T[0,-1]=1
            T[-1,0]=1
    elif mode=="random":
        T=np.random.rand(n,n)
        for i in range(0,n):
            for j in range(i,n):
                T[i,j]=T[j,i]
    elif mode=="standard" or mode=="Hamming":
        ispowertwo=(n & (n-1) == 0) and n !=0
        if not ispowertwo:
            raise ValueError('mode "'+mode+'" needs n to be a power of two')
#def T_full_standard_mixer(n,d=1):
        T=np.zeros((n,n))
        log2n=int(np.log2(n))
        for i in range(n):
            for j in range(n):
                s1="{0:b}".format(i).zfill(log2n)
                s2="{0:b}".format(j).zfill(log2n)
                if hamming(s1,s2)==d:
                    T[i,j]=1
    else:
        raise NotImplementedError('mode "'+mode+'" not implemented')

    return T

def T_sum_Hamming_distance(T):
    val=0
    val2=0
    n=T.shape[0]
    for i in range(n):
        for j in range(n):
            if not math.isclose(T[i,j],0,abs_tol=1e-7):
                s1="{0:b}".format(i).zfill(n)
                s2="{0:b}".format(j).zfill(n)
                tmp=hamming(s1,s2)
                if tmp>1:
                    val2+=tmp
                else:
                    val2+=tmp/10
                val += tmp
    return val, val2


class PauliStringTP:
    def __init__(self, excludeI=False):
        self.excludeI=excludeI
        self.items=[]
    def get_items_PS(self,tp):
        if isinstance(tp,Pauli):
            if not self.excludeI:
                self.items.append(tp)
            elif not tp==I:
                self.items.append(tp)
        else:
            tpL,tpR=tp.args
            if isinstance(tpL, TensorProduct):
                self.get_items_PS(tpL)
            else:
                if not self.excludeI:
                    self.items.append(tpL)
                elif not tpL==I:
                    self.items.append(tpL)
            if isinstance(tpR, TensorProduct):
                self.get_items_PS(tpR)
            else:
                if not self.excludeI:
                    self.items.append(tpR)
                elif not tpR==I:
                    self.items.append(tpR)

def HtoString(H, symbolic=False):
    ret=''
    for item in H.args:### go through all items of the sum (Pauli strings)
        if isinstance(item, Mul):### remove float
            if symbolic:
                fval,_,item = item.args
            else:
                if len(item.args)>2:
                    fval,tmp,item = item.args
                    if not math.isclose(fval,0,abs_tol=1e-7):
                        raise AssertionError("Encountered imaginary part that is not close to zero, aborting!", fval, tmp, item)
                else:
                    fval,item = item.args
                    if math.isclose(fval,0,abs_tol=1e-7):
                        item=None
                        print("depug: close to zero", fval, item)
            ret+=f'{fval:+.2f}'+" "
        if isinstance(item, TensorProduct) or isinstance(item, Pauli):### go through Pauli string
            tps=PauliStringTP()
            tps.get_items_PS(item)
            for p in tps.items:
                if p==1:
                    ret+="I"
                if p==X:
                    ret+="X"
                if p==Y:
                    ret+="Y"
                if p==Z:
                    ret+="Z"
        ret+=" "
    return ret

def num_Cnot(H, symbolic=False):
    sqg=0
    cnot=0
    for item in H.args:### go through all items of the sum (Pauli strings)
#         print(type(item))
        if isinstance(item, Mul):### remove float
            if symbolic:
                fval,_,item = item.args
            else:
                if len(item.args)>2:
                    fval,tmp,item = item.args
                    if not math.isclose(fval,0,abs_tol=1e-7):
                        raise AssertionError("Encountered imaginary part that is not close to zero, aborting!", fval, tmp, item)
                else:
                    fval,item = item.args
                    if math.isclose(fval,0,abs_tol=1e-7):
                        item=None
                        print("depug: close to zero", fval, item)
        if isinstance(item, TensorProduct) or isinstance(item, Pauli):### go through Pauli string
            tps=PauliStringTP(excludeI=True)
            tps.get_items_PS(item)
            tmp=len(tps.items)
            if tmp==1:
                sqg+=1
            elif tmp>1:
                cnot+=2*(tmp-1)
    return sqg,cnot

def get_g(binstrings):
    n=len(binstrings[0])
    x=binsymbols('x:'+str(n))
    expr=1
    for bs in binstrings:
        tmp_expr=0
        for i in range(n):
            if bs[i]=='0':
                tmp_expr+=x[i]
            else:
                tmp_expr+=(x[i]-1)**2
        expr*=tmp_expr
    return x, expand(expr)


def convert_to_ps(bs1, bs2):
    n=len(bs1)

    for j in range(n):
        if bs1[j]=="1" and bs2[j]=="0":
            tmp=1/2*(X-1j*Y)
        elif bs1[j]=="0" and bs2[j]=="1":
            tmp=1/2*(X+1j*Y)
        elif bs1[j]=="1" and bs2[j]=="1":
            tmp=1/2*(I-Z)
        else:# bs[j]=="0" and bs[j]=="0":
            tmp=1/2*(I+Z)
        if j==0:
            pauli_str=tmp
        else:
            pauli_str=TensorProduct(pauli_str,tmp)
    return pauli_str

def get_overlap(binstringsA, binstringsB):
    overlap=[]
    mA=len(binstringsA)
    mB=len(binstringsB)
    for i in range(mA):
        for j in range(mB):
            if binstringsA[i] == binstringsB[j]:
                overlap.append(binstringsA[i])
    return overlap

def get_minus(binstringsA, binstringsB):
    minus=[]
    indices=[]
    mA=len(binstringsA)
    mB=len(binstringsB)
    for i in range(mA):
        found=False
        for j in range(mB):
            if binstringsA[i] == binstringsB[j]:
                found=True
                break
        if not found:
            minus.append(binstringsA[i])
            indices.append(i)
    return minus, indices

def add(binstringsA, binstringsB):
    add=binstringsA.copy()
    mA=len(binstringsA)
    mB=len(binstringsB)
    for i in range(mB):
        found=False
        for j in range(mA):
            if binstringsA[j] == binstringsB[i]:
                found=True
                break
        if not found:
            add.append(binstringsB[i])
    return add

def get_bitwise_negated_strings(binstrings):
    n=len(binstrings[0])
    m=len(binstrings)
    
    binstrings_neg = []
    for i in range(m):
        tmp=''
        for j in range(n):
            if binstrings[i][j]=="1":
                tmp+="0"
            else:
                tmp+="1"
        binstrings_neg.append(tmp)
    return binstrings_neg

def get_negated_strings(binstrings, mask):
    n=len(binstrings[0])
    m=len(binstrings)
    
    binstrings_neg = []
    for i in range(m):
        tmp=''
        for j in range(n):
            if mask[j]=='1':
                if binstrings[i][j]=="1":
                    tmp+="0"
                else:
                    tmp+="1"
            else:
                tmp+=binstrings[i][j]
        binstrings_neg.append(tmp)
    return binstrings_neg


def get_Pauli_string(binstrings, T, symbolic=False):
    m=len(binstrings)
   
    pauli_str=0
    if symbolic:
        for i in range(m):
            for j in range(m):
                tmp_ps = convert_to_ps(binstrings[i], binstrings[j])
                pauli_str+=T[i,j]*tmp_ps
    else:
        for i in range(m):
            for j in range(m):
                if not math.isclose(T[i,j],0,abs_tol=1e-7):
                    tmp_ps = convert_to_ps(binstrings[i], binstrings[j])
                    pauli_str+=T[i,j]*tmp_ps

    return pauli_str

def simplifyH(H):
    for i in range(10):
        H = H.expand(tensorproduct=True)
    H=evaluate_pauli_product(H)
    return H

def get_H(stringlist,T,simplify=True, symbolic=False, verbose=False):
    H=get_Pauli_string(stringlist, T, symbolic=symbolic)
    if simplify:
        H=simplifyH(H)
    if verbose:
        print("#sqg, #cnots=",num_Cnot(H, symbolic=symbolic))
    return H

def print_info(stringlist,T,disp=True,simplify=True, disp_d=True, disp_g=True,disp_m=True, disp_H=True, disp_neg=True):
    n=len(stringlist[0])
    m=len(stringlist)

    if disp_d:
        print("Hamming distance=", T_sum_Hamming_distance(T))
    
    H=get_Pauli_string(stringlist, T)
    if simplify:
        H=simplifyH(H)
#     T+=eye(len(stringlist))*d
#     m = Matrix(T)
#     P,D = m.diagonalize()
#     display(P)
#     display(D)
    if disp_H:
        display("H=", H)
    print("#sqg, #cnots=",num_Cnot(H))

    if m<2**n:
        stringlist_neg=get_bitwise_negated_strings(stringlist)
        overlap = get_overlap(stringlist,stringlist_neg)
        if not len(overlap)==0:
            print("overlap with bitwise negated stringlist =", overlap)
        else:
            H_neg=get_Pauli_string(stringlist_neg, T)+H
            if simplify:
                H_neg=simplifyH(H_neg)
            if disp_neg:
                display("H+H_neg=", H_neg)
                print("#sqg, #cnots=",num_Cnot(H_neg))

        minus, m_ind = get_minus(stringlist_neg,stringlist)
        if len(minus)<2:
            print("minus =", minus)
        else:
            S = T[m_ind].T[m_ind].T ## .T is the transpose
            H_minus=get_Pauli_string(minus, S)+H
            if simplify:
                H_minus=simplifyH(H_minus)
            if disp_m:
                display("H+H_minus=", H_minus)
                print("#sqg, #cnots=",num_Cnot(H_minus))
    
        if disp_g:
            x, pen = get_g(stringlist)

            g=lambdify(x, pen, "numpy")

            print("g(x)=",pen)              
            print("is zero for the following bitstrings:")
            l=len(stringlist[0])
            if l==1:
                for y1 in [0,1]:
                    val = g(y1)
                    if val<0:
                        print(y1,"=",val)
                    if val==0:
                        print(y1)
            elif l==2:
                for y1 in [0,1]:
                    for y2 in [0,1]:
                        val = g(y1,y2)
                        if val<0:
                            print(y1,y2,"=",val)
                        if val==0:
                            print(y1,y2)
            elif l==3:
                for y1 in [0,1]:
                    for y2 in [0,1]:
                        for y3 in [0,1]:
                            val = g(y1,y2,y3)
                            if val<0:
                                print(y1,y2,y3,"=",val)
                            if val==0:
                                print(y1,y2,y3)


def print_info2(stringlist,T,simplify=True):
    n=len(stringlist[0])
    m=len(stringlist)

    all_strings=stringlist.copy()

    H=get_Pauli_string(stringlist, T)
    print("m,2**n",m,2**n)
    
    if m<2**n:
        first=True
        for mask in reversed(all_states(n)):
            if first:
                first=False
                #continue
            stringlist_neg = get_negated_strings(stringlist,mask)
            minus, m_ind = get_minus(stringlist_neg,all_strings)
            if len(minus)<2:
                print("minus =", minus)
            else:
                print("add minus =", minus)
                S = T[m_ind].T[m_ind].T ## .T is the transpose
                H+=get_Pauli_string(minus, S)
                all_strings = add(all_strings, minus)
                break

    if simplify:
        H=simplifyH(H)

    display("H=", H)
    print("#sqg, #cnots=",num_Cnot(H))
