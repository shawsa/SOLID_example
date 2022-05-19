from abc import ABC, abstractmethod
from grid import Grid
import numpy as np
from typing import Callable

class BoundaryCondition(ABC):

    @abstractmethod
    def is_forcing_valid(self, forcing: Callable, grid: Grid) -> bool:
        '''Validate that the problem is well posed.'''
        raise NotImplementedError

    @abstractmethod
    def left_stencil(self, grid: Grid):
        '''The left stencil to be used in the Laplacian operator.'''
        raise NotImplementedError

    @abstractmethod
    def right_stencil(self, grid: Grid):
        '''The left stencil to be used in the Laplacian operator.'''


class DirichletBC(BoundaryCondition):

    def is_forcing_valid(self, forcing: Callable, grid: Grid) -> bool:
        '''Steady state is always well posed. Verify that the 
        boundaries are not forced.'''
        return (abs(forcing(grid.a)) < 1e-10) and \
               (abs(forcing(grid.b)) < 1e-10)

    def left_stencil(self, grid: Grid):
        row = np.zeros_like(grid.xs_ghost)
        row[1] = 1
        return row

    def right_stencil(self, grid: Grid):
        row = np.zeros_like(grid.xs_ghost)
        row[-2] = 1
        return row

class NeumannBC(BoundaryCondition):

    def is_forcing_valid(self, forcing: Callable, grid: Grid) -> bool:
        '''Steady state requires net flux to be zero.'''
        return abs(grid.trap_rule(forcing)) < 1e-10

    def left_stencil(self, grid: Grid):
        row = np.zeros_like(grid.xs_ghost)
        row[:3] = np.array([-1, 0, 1]) / (2*grid.spacing)
        return row

    def right_stencil(self, grid: Grid):
        row = np.zeros_like(grid.xs_ghost)
        row[-3:] = np.array([-1, 0, 1]) / (2*grid.spacing)
        return row
