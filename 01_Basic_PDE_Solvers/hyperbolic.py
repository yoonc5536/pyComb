import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

'''
1: First order Upwind
2: Midpoint Leapfrog
3: Lax method
4: Euler Implicit
5: Lax-Wendroff method
6: Two-Step Lax-Wendroff Method
7: MacCormick Method
8: Crank-Nicolson Method
'''

solverType = 8

jm = 401
nm = 251
dt = 0.004
domainLength = 400.0
c = 250.0
pi = 4*np.arctan(1)

x = np.linspace(0, domainLength, jm)
dx = x[1]-x[0]
cfl = c*dt/dx

un = np.zeros(jm)
un[50:110] = 100*np.sin(pi*(x[50:110]-50)/60)
u = un.copy()

sls = slice(1,-1)

# exact solution
ue = np.zeros_like(u)
for n in range(1,nm):
    t = dt*n
    xm1 = 50+c*t
    xm2 = 110 + c*t
    for j in range(jm):
        if (x[j]<xm1):
            ue[j] = 0
        elif (x[j]<xm2):
            ue[j] = 100*np.sin(pi*(x[j]-c*t-50)/60)
        else:
            ue[j] = 0

# numerical solution
if solverType == 1:
    for n in range(1,nm):
        u[sls] = u[sls] + cfl*(u[:-2]-u[sls])
if solverType == 2:
    unm = np.zeros_like(u)    
    unm[sls] = un[sls] - cfl*(un[:-2]-un[sls])
    for n in range(1,nm):
        u[sls] = unm[sls] + cfl*(un[:-2]-un[2:])
        unm = un.copy()
        un = u.copy()
if solverType == 3:
    for n in range(1,nm):
        u[sls] = (u[2:]+u[:-2])/2.0+cfl/2.0*(u[:-2]-u[2:])
if solverType == 4:
    for n in range(1,nm):
        arr = np.ones(jm-2)
        matA = spdiags([-arr*0.5*cfl, arr, 0.5*cfl*arr], [-1, 0, 1], jm-2, jm-2)
        matB = u[sls]
        matB[0] += 0.5*cfl*0
        matB[jm-3] -= 0.5*cfl*0
        u[sls] = spsolve(matA.tocsr(), matB)
if solverType == 5:
    up = np.zeros_like(u)
    for n in range(1,nm):
        up[sls] = u[sls] - cfl/2.0*(u[2:]-u[:-2])
        u[sls] = up[sls] + cfl*cfl/2.0*(u[2:]-2*u[sls]+u[:-2])
if solverType == 6:
    up = np.zeros_like(u)
    for n in range(1,nm):        
        up[sls] = (u[2:]+u[sls])/2.0 - cfl/2.0*(u[2:]-u[sls])
        u[sls] = u[sls] - cfl*(up[sls]-up[:-2])
if solverType == 7:
    up = np.zeros_like(u)
    for n in range(1,nm):            
        up[sls] = u[sls] - cfl*(u[2:]-u[sls])
        u[sls] = 0.5*(u[sls]+up[sls]-cfl*(up[sls]-up[:-2]))
if solverType == 8:
    for n in range(1,nm):
        arr = np.ones(jm-2)
        matA = spdiags([-arr*0.25*cfl, arr, 0.25*cfl*arr], [-1, 0, 1], jm-2, jm-2)
        matB = u[sls]-0.25*cfl*(u[2:]-u[:-2])
        matB[0] += 0.25*cfl*0
        matB[jm-3] -= 0.25*cfl*0
        u[sls] = spsolve(matA.tocsr(), matB)
        
plt.plot(x, u)
plt.show()