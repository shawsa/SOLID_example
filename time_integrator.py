'''A time-integration class for solving ODEs numerically.'''

from abc import ABC, abstractmethod, abstractproperty
from time_domain import TimeDomain, TimeDomain_Start_Stop_Steps

class TimeIntegrator(ABC):

    @abstractproperty
    def name(self):
        raise NotImplementedError
        

    @abstractmethod
    def solution_generator(self, u0, rhs, time: TimeDomain):
        raise NotImplementedError

    def solve(self, u0, rhs, time: TimeDomain):
        return time.array, list(self.solution_generator(u0, rhs, time))

    def t_final(self, u0, rhs, time: TimeDomain):
        for u in self.solution_generator(u0, rhs, time):
            pass
        return u


class SingleStepMethod(TimeIntegrator):

    @abstractmethod
    def update(self, t, u, f, h):
        raise NotImplementedError

    def solution_generator(self, u0, rhs, time: TimeDomain):
        u = u0
        yield u
        for t in time.array[:-1]:
            u = self.update(t, u, rhs, time.spacing)
            yield u


class Euler(SingleStepMethod):

    @property
    def name(self):
        return 'Euler'

    def update(self, t, u, f, h):
        return u + h*f(t, u)

class RK4(SingleStepMethod):
    @property
    def name(self):
        return 'RK4'

    def update(self, t, u, f, h):
        k1 = f(t, u)
        k2 = f(t+h/2, u + h/2*k1)
        k3 = f(t+h/2, u + h/2*k2)
        k4 = f(t+h, u + h*k3)
        return u + h/6*(k1 + 2*k2 + 2*k3 + k4)

class AB2(TimeIntegrator):
    def __init__(self, seed: TimeIntegrator, seed_steps):
        self.seed = seed
        self.seed_steps = seed_steps

    @property
    def name(self):
        return f'AB2 (seed: {self.seed.name})'


    def update(self, t, y, y_old, f, h):
        return y + 3/2*h*f(t, y) - 1/2*h*f(t-h, y_old)

    def solution_generator(self, u0, rhs, time: TimeDomain):
        u = u0
        t = time.start
        yield u
        u_old = u
        
        seed_time = TimeDomain_Start_Stop_Steps(
                        time.start,
                        time.spacing,
                        self.seed_steps)

        u = self.seed.t_final(u, rhs, seed_time)
        t = time.start + time.spacing
        yield u
        for t in time.array[1:-1]:
            u, u_old = self.update(t, u, u_old, rhs, time.spacing), u
            yield u
