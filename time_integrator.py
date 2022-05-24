'''A time-integrator class for solving ODEs.'''

from abc import ABC, abstractmethod, abstractproperty

from time_domain import TimeDomain, TimeDomain_Start_Stop_Steps


class TimeIntegrator(ABC):
    @abstractmethod
    def solution_generator(self, u0, rhs, time: TimeDomain):
        raise NotImplementedError

    def solve(self, u0, rhs, time: TimeDomain):
        sol_gen = self.solution_generator(u0, rhs, time)
        return time.array, list(sol_gen)

    def t_final(self, u0, rhs, time: TimeDomain):
        sol_gen = self.solution_generator(u0, rhs, time)
        for u in sol_gen:
            pass
        return u

    @abstractproperty
    def name(self):
        raise NotImplementedError


class EulerIntegrator(TimeIntegrator):

    @staticmethod
    def solution_generator(u0, rhs, time: TimeDomain):
        u = u0
        yield u
        for t in time.array[:-1]:
            u = u + time.spacing*rhs(t, u)
            yield u

    @property
    def name(self):
        return 'Euler'


class RK4_Integrator(TimeIntegrator):
    @staticmethod
    def solution_generator(u0, rhs, time: TimeDomain):
        u = u0
        yield u
        for t in time.array[:-1]:
            k1 = rhs(t, u)
            k2 = rhs(t+time.spacing/2, u + time.spacing/2*k1)
            k3 = rhs(t+time.spacing/2, u + time.spacing/2*k2)
            k4 = rhs(t+time.spacing, u + time.spacing*k3)
            u = u + time.spacing/6*(k1 + 2*k2 + 2*k3 + k4)
            yield u

    @property
    def name(self):
        return 'RK4'


class AB2_Integrator(TimeIntegrator):
    def __init__(self, seed: TimeIntegrator, seed_step):
        self.seed = seed
        self.seed_step = seed_step

    def solution_generator(self, u0, rhs, time: TimeDomain):
        u = u0
        u_old = u
        seed_domain = TimeDomain_Start_Stop_Steps(
                            start=time.start,
                            end=time.spacing,
                            steps=self.seed_step)
        u = self.seed.t_final(u0, rhs, seed_domain)
        yield u
        for t in time.array[1:-1]:
            u, u_old = (u +
                        3/2*time.spacing*rhs(t, u) +
                        -1/2*time.spacing*rhs(t-time.spacing, u_old)), u
            yield u

    @property
    def name(self):
        return f'AB2 (seed: {self.seed.name})'
