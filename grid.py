import numpy as np
from scipy.linalg import circulant


class Grid:
    def __init__(self, a, b, N):
        self.a = a
        self.b = b
        self.N = N
        self.spacing = (b - a)/(N-1)
        self.initialize_xs()

    def initialize_xs(self):
        self.xs_ghost = np.linspace(self.a - self.spacing,
                                    self.b + self.spacing,
                                    self.N + 2)
        self.xs = self.xs_ghost[1:-1]

    def trap_rule(self, func):
        return (
                   (func(self.a) + func(self.b))/2 +
                   np.sum(func(self.xs[1:-1]))
               ) / self.spacing

    def lapacian(self):
        '''In 1D, the second order approximation applies the stencil
        [1 -2 1] to each point on the interior of the grid. The
        circulant function gives us this if we ignore the first
        two rows.'''
        data = [1, -2, 1] + [0]*(self.N-3 + 2)
        circ = circulant(data)[2:]
        return circ / self.spacing**2

