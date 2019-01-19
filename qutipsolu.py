import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()

#identity and pauli matrices
I     = qt.qeye(2)
X     = qt.sigmax()
Y     = qt.sigmay()
Z     = qt.sigmaz()
sigma = np.array([I, X, Y, Z])

#time array
t0    = 0.001 # not 0 so we can vary parameters with 1/t without getting infinities
tf    = 2.001
steps = 20000
times = np.linspace(t0,tf,steps)
ds    = (tf - t0)/steps

#noise operators
V1 = X
V2 = Y
V3 = Z

def V_coeff(t,args): #the time varying part of V
        return np.sqrt(0.5)*(t)
    
noise = [[V1, V_coeff], [V2, V_coeff], [V3, V_coeff]]

#hamiltonian
H1 = I

def H_coeff(t, args): # the time varying part of H
    return 1/t
    
H = [[H1, H_coeff]]

#E0
A0 = 0.5*(I + X)
B0 = 0.5*(I + Y)

a  = 1
b  = 1
psi0 = np.array([a*qt.basis(2,0), b*qt.basis(2,1)])
psi1 = X.eigenstates()[1]
psi2 = Y.eigenstates()[1]
psi3 = Z.eigenstates()[1]
psi = [psi0, psi1, psi2, psi3]

#solving the master equation
def x():
    xt =[]
    x = [qt.mesolve(H, psi[0][1], times, noise, [A0]).expect[0] + 
          qt.mesolve(H, psi[0][0], times, noise, [A0]).expect[0]]
    for i in range(3):
        i += 1
        x.append(qt.mesolve(H, psi[i][1], times, noise, [A0]).expect[0] - 
                 qt.mesolve(H, psi[i][0], times, noise, [A0]).expect[0])
    for i in range(len(times)):
        xt.append(np.asarray([x[0][i], x[1][i], x[2][i], x[3][i]]))
    return xt
       
def y():
    yt =[]
    y = [qt.mesolve(H, psi[0][0], times, noise, [B0]).expect[0] + 
          qt.mesolve(H, psi[0][1], times, noise, [B0]).expect[0]]
    for i in range(3):
        i += 1
        y.append(qt.mesolve(H, psi[i][1], times, noise, [B0]).expect[0] - 
                 qt.mesolve(H, psi[i][0], times, noise, [B0]).expect[0])
    for i in range(len(times)):
        yt.append(np.asarray([y[0][i], y[1][i], y[2][i], y[3][i]]))
    return yt

#degree of incompatibility
def comp(x,y):
    i = 0
    c = []
    while (i < len(x)):
        c.append(np.linalg.norm(x[i] - y[i]) + np.linalg.norm(x[i] + y[i]) - 2)
        i = i + 1        
    return c


def samesign(a,b):
    return a * b > 0

#finding decoherence time
def bisect(ctemp):
    midpoint = int(len(times)/2)
    low = 0
    high = len(times)
    n = int((np.log(tf-t0)-np.log(ds))/np.log(tf)) + 1
    for i in range(n):
        midpoint = int((low + high)/2)
        if samesign(ctemp[low], ctemp[midpoint]):
            low = midpoint
        else:
            high = midpoint
    return midpoint

#graphs
plt.clf()
xt = x()
yt = y()
ctemp = comp(xt, yt)
#ctempo = comp(xt, yt)
tc = times[bisect(ctemp)]
print(tc, "+ or - ", ds)
plt.plot(times, ctemp)
plt.grid()
plt.axis([(t0),(tf),(min(ctemp)),(max(ctemp))])
plt.xlabel('Time (s)')
plt.ylabel('Degree of Incompatibility')
#Plotting the region of compatibility with a green tint
p = plt.axhspan(0, -10, facecolor='#2ca02c', alpha=0.3)
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))