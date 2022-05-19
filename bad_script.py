'''Solve the time-dependent heat equation and compare to steady-state solution.'''
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy.linalg import circulant
from tqdm import tqdm

# bcs = 'Neumann'
bcs = 'Dirichlet'

N = 201 # number of points
a, b = -np.pi, np.pi
h = (b-a)/(N-1)

xs_ghost = np.linspace(a-h, b+h, N+2)
xs = xs_ghost[1:-1]

if bcs == 'Dirichlet':
    f = np.sin(2*xs)
elif bcs == 'Neumann':
    f = np.cos(3*xs)

M = np.zeros((N+2, N+2))
M[1:-1] = circulant([1, -2, 1] + [0]*(N-3+2))[2:] / h**2
if bcs == 'Neumann':
    M[0, :3] = np.array([-1, 0, 1]) / (2*h)
    M[-1, -3:] = np.array([-1, 0, 1]) / (2*h)

u0 = np.zeros_like(xs)
us = np.zeros_like(xs_ghost)
us[1:-1] = u0
if bcs == 'Dirchlet':
    us[1] = 0
    us[-2] = 0
elif bcs == 'Neumann':
    us[0] = us[2]
    us[-1] = us[-3]

if bcs == 'Dirchlet':
    assert abs(f[0]) < 1e-10
    assert abs(f[-1]) < 1e-10
elif bcs == 'Neumann':
    integral  = (.5*f[0] + np.sum(f[1:-1]) + .5*f[-1])/h
    assert abs(integral) < 1e-10

dt = 1e-4
steps = 8_000

uf = np.zeros_like(xs_ghost)
if bcs == 'Dirichlet':
    uf[1:-1] = la.solve(-M[1:-1, 1:-1], f)
elif bcs == 'Neumann':
    M2 = M.copy()
    M2[-1] = np.ones_like(xs_ghost)
    M2[-1, 0] = .5
    M2[-1, -1] = .5
    rhs = np.zeros_like(xs_ghost)
    rhs[0] = 0
    rhs[1:-1] = f
    rhs[-1] = 0
    uf[1:-1] = la.solve(-M2, rhs)[1:-1]


us = np.zeros_like(xs_ghost)
us[1:-1] = u0

plt.ion()
plt.plot(xs, f, 'r:', label='forcing')
plt.plot(xs, uf[1:-1], 'g--', label='Steady State')
u_line, = plt.plot(xs, us[1:-1], 'b-', label='$T$')
plt.legend()
plt.ylim(-2, 2)
plt.xlim(a, b)
plt.show()

for step in tqdm(range(steps)):
    us[1:-1] += dt * ((M@us)[1:-1] + f)
    if bcs == 'Dirichlet':
        us[1] = 0
        us[-2] = 0
    elif bcs == 'Neumann':
        us[0] = us[2]
        us[-1] = us[-3]
    if step%40 ==0:
        u_line.set_ydata(us[1:-1])
        plt.pause(dt)
