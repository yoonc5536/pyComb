import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

solverType = 'C-N'

dt = 0.002      # time step
v = 0.000217    # viscosity
u0 = 40         # flat plate speed (m/s)
jm = 41         # grid size
nm = 1000        # total time step number
nimg = 100
beta = 0.8

p = np.linspace(0, 0.04, jm)
dy = p[2]-p[1]

cfl = (v*dt)/dy/dy

u=np.zeros(jm)

# Boundary Condition
u[0] = u0
u[jm-1] = 0
sls = slice(1,-1)
un = u.copy()
unm = un.copy()

for n in range(1,nm-1):
    print('tiem step = %d' % n)
    unm = un.copy()
    un = u.copy()
    if solverType == 'FTCS':
        u[0], u[jm-1], u[sls] = u0, 0, un[sls] + cfl*(un[2:]-2*un[sls]+un[:-2])
    elif solverType == 'D-F':
        coeff1, coeff2, coeff3 = 1+2*cfl, 1-2*cfl, 2*cfl
        u[sls] = coeff2*unm[sls]+coeff3*(un[2:]+un[:-2])
        u[0], u[jm-1], u[sls] = u0, 0, u[sls]/coeff1
    elif solverType == "IMP":
        arr = np.ones(jm-2)
        matA = spdiags([arr*cfl, -arr*(1.0+2.0*cfl), arr*cfl], [-1, 0, 1], jm-2, jm-2)
        matB = -u[sls]
        matB[0] += -cfl*u0
        matB[jm-3] += 0
        u[0], u[jm-1], u[sls] = u0, 0, spsolve(matA.tocsr(), matB)
    elif solverType == "C-N":
        arr = np.ones(jm-2)
        matA = spdiags([-arr*cfl, arr*(2.0+2.0*cfl), -arr*cfl], [-1, 0, 1], jm-2, jm-2)
        matB = cfl*(u[2:]-2*u[sls]+u[:-2])+2*u[sls]
        matB[0] += cfl*u0
        matB[jm-3] += 0
        u[0], u[jm-1], u[sls] = u0, 0, spsolve(matA.tocsr(), matB)
    if n%nimg == 0:
        plt.plot(u, p)
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Distance (m)')
        plt.savefig('profile_'+'{:03d}'.format(n)+'.png')
        plt.clf()

