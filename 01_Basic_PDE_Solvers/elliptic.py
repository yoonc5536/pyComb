import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

'''
1: Point Jacobi
2: Point Gauss-Seidel
3: Point Successive Over-Relaxation Method
4: Line Gauss-Seidel (x-sweep)
5: Line Gauss-Seidel (y-sweep)
6: Line Successive Over-Relaxation Method (x-sweep)
7: Line Successive Over-Relaxation Method (y-sweep)
8: ADI
'''
solverType = 8

pi = 4*np.arctan(1)
im, jm = 21, 41
T1, T2, T3, T4 = 100,0,0,0
domainLength, domainHeight = 1.0, 1.0
relax = 1.1
nIter = 1000

# define mesh
x0 = np.linspace(0, domainLength, im)
y0 = np.linspace(0, domainHeight, jm)
dx = x0[1]-x0[0]; dy = y0[1] - y0[0]
beta = dx/dy

x = np.zeros((jm, im))
y = np.zeros_like(x).transpose()
Texact = np.zeros_like(x)
T = np.zeros_like(Texact)
T[0] = T1

# evaluate exact solution
x[:], y[:] = x0, y0
y = np.transpose(y)

for n in range(1,50):
    clambda = n*pi/domainLength;
    coeffa = -200./n/pi*(np.cos(n*pi)-1.)/(1.-np.exp(2.*clambda*domainHeight))
    coeffb = np.exp(clambda*y)-np.exp(2.*clambda*domainHeight-clambda*y)
    Texact = Texact+coeffa*np.sin(clambda*x)*coeffb

sls = slice(1,-1)    
for n in range(nIter):
    if solverType == 1:
        Tn = T.copy()
        coeff1=1./2./(1.+beta*beta)
        T[sls, sls] = coeff1*(Tn[sls, 2:]+Tn[sls, :-2]+beta*beta*(Tn[2:, sls]+Tn[:-2, sls]))
    elif solverType == 2:
        coeff1=1./2./(1.+beta*beta)
        T[sls, sls] = coeff1*(T[sls, 2:]+T[sls, :-2]+beta*beta*(T[2:, sls]+T[:-2, sls]))
    elif solverType == 3:
        coeff1=relax/2./(1.+beta*beta)
        T[sls, sls] = T[sls, sls]+coeff1*(T[sls, 2:]+T[sls, :-2]+beta*beta*(T[2:, sls]+T[:-2, sls]) - 2*(1+beta*beta)*T[sls, sls])
        pass
    elif solverType == 4:
        arr = np.ones(im-2)
        matA = spdiags([arr, -2*arr*(1+beta*beta), arr], [-1, 0, 1], im-2, im-2)
        for j in range(1, jm-2):
            matB = -beta*beta*(T[j+1, sls]+T[j-1, sls])
            matB[0] -= T2
            matB[im-3] -= T4
            T[j, sls] = spsolve(matA.tocsr(), matB)
    elif solverType == 5:
        arr = np.ones(jm-2)
        matA = spdiags([arr*beta*beta, -2*arr*(1+beta*beta), arr*beta*beta], [-1, 0, 1], jm-2, jm-2)
        for i in range(1, im-2):
            matB = -T[sls, i+1]-T[sls, i-1]
            matB[0] -= beta*beta*T1
            matB[jm-3] -= beta*beta*T3
            T[sls, i] = spsolve(matA.tocsr(), matB)
    elif solverType == 6:
        arr = np.ones(im-2)
        matA = spdiags([relax*arr, -2*arr*(1+beta*beta), relax*arr], [-1, 0, 1], im-2, im-2)
        for j in range(1, jm-2):
            matB = -relax*beta*beta*(T[j+1, sls]+T[j-1, sls])
            matB -= (1-relax)*(2.0*(1.0+beta*beta))*T[j, sls]
            matB[0] -= relax*T2
            matB[im-3] -= relax*T4
            T[j, sls] = spsolve(matA.tocsr(), matB)
    elif solverType == 7:
        arr = np.ones(jm-2)
        matA = spdiags([relax*beta*beta*arr, -2*arr*(1+beta*beta), relax*beta*beta*arr], [-1, 0, 1], jm-2, jm-2)
        for i in range(1, im-2):
            matB = -relax*(T[sls, i+1]+T[sls, i-1])
            matB -= (1-relax)*(2.0*(1.0+beta*beta))*T[sls, i]
            matB[0] -= relax*beta*beta*T1
            matB[jm-3] -= relax*beta*beta*T3
            T[sls, i] = spsolve(matA.tocsr(), matB)
    elif solverType == 8:
        arr = np.ones(im-2)
        matA = spdiags([arr, -2*arr*(1+beta*beta), arr], [-1, 0, 1], im-2, im-2)
        for j in range(1, jm-2):
            matB = -beta*beta*(T[j+1, sls]+T[j-1, sls])
            matB[0] -= T2
            matB[im-3] -= T4
            T[j, sls] = spsolve(matA.tocsr(), matB)
        arr = np.ones(jm-2)
        matA = spdiags([arr*beta*beta, -2*arr*(1+beta*beta), arr*beta*beta], [-1, 0, 1], jm-2, jm-2)
        for i in range(1, im-2):
            matB = -T[sls, i+1]-T[sls, i-1]
            matB[0] -= beta*beta*T1
            matB[jm-3] -= beta*beta*T3
            T[sls, i] = spsolve(matA.tocsr(), matB)
        

plt.contourf(x0, y0, T)
plt.show()



