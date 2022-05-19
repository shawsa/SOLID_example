from boundary_condition import BoundaryCondition
from grid import Grid
import numpy as np
from typing import Callable

class Model:
    def __init__(self, 
                 forcing: Callable,
                 bc: BoundaryCondition,
                 grid: Grid):
        self.forcing = forcing
        self.bc = bc
        self.grid = grid

        assert bc.is_forcing_valid(forcing, grid)

        self.f = forcing(grid.xs)
        self.initialize_operator()

    def initialize_operator(self):
        self.op = np.zeros((self.grid.N+2, self.grid.N+2))
        self.op[1:-1] = self.grid.lapacian()
        self.op[0] = self.bc.left_stencil(self.grid)
        self.op[-1] = self.bc.right_stencil(self.grid)

    def steady_state(self):

        u0 = np.zeros_like(xs)
        us = np.zeros_like(xs_ghost)
        us[1:-1] = u0
        if bcs == 'Dirchlet':
            us[1] = 0
            us[-2] = 0
        elif bcs == 'Neumann':
            us[0] = us[2]
            us[-1] = us[-3]
