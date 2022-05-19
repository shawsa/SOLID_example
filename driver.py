'''Solve the time-dependent heat equation and compare to steady-state solution.'''
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from tqdm import tqdm

from boundary_condition import DirichletBC
from grid import Grid
from model import Model


def main():

    grid = Grid(-np.pi, np.pi, 201)
    bc = DirichletBC()
    def forcing(xs):
        return np.sin(2*xs)

    model = Model(forcing, bc, grid)

    uf = model.steady_state()


    # bcs = 'Neumann'
    # def forcing(xs):
    #     return np.sin(3*xs)

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

if __name__ == '__main__':
    main()
