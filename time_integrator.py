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

    # This solve function is different that the old solve function.
    # It captures the behavior when ret='all' and returns the 
    # full solution over time. This is an example of the "Strategy
    # Pattern" when the unit is a function rather than a class.
    # When different functionality is used, a differetn function is called
    # rather than a different parameter being passed.
    def solve(self, u0, rhs, time: TimeDomain):
        return time.array, list(self.solution_generator(u0, rhs, time))

    def t_final(self, u0, rhs, time: TimeDomain):
        for u in self.solution_generator(u0, rhs, time):
            pass
        return u

# A Single step method is a kind of time-integrator that has a specific
# pattern it follows to generate a solution. This is an example of 
# "Dependency Injection". The solution_generator uses the "update" function
# but the update function is selected by the choice of sub-class. This 
# method inverts the dependency, so the update function depends on the 
# solution generator instead.
class SingleStepMethod(TimeIntegrator):
    @abstractmethod
    def update(self, t, u, f, h):
        raise NotImplementedError

    # Notice here, that this implementation is the essence
    # of a single-step method. When we say "single-step method', this
    # looping and updating is what we mean. This class is identical
    # to our mathematical abstraction.
    # Here we've used the "Generator Pattern" to decouple the looping
    # behavior from the update behavior.
    def solution_generator(self, u0, rhs, time: TimeDomain):
        u = u0
        yield u
        for t in time.array[:-1]:
            u = self.update(t, u, rhs, time.spacing)
            yield u


class Euler(SingleStepMethod):
    # This class encodes much more of the meaning of forward Euler than the
    # old script. The old script said that it was just the update function, but
    # this isn't exactly correct. The essential quality is the update function,
    # but it is a kind of single step method, which has a specific looping behavior,
    # and it is a kind of Time-Integrator which numerically solves (this is ambiguous, what
    # does it return?) an ODE. This hierarchy clearly delinates all of the lines of code
    # that we would say comprised Euler's method in the old script.
    @property
    def name(self):
        return 'Euler'

    def update(self, t, u, f, h):
        return u + h*f(t, u)

class RK4(SingleStepMethod):
    # Here again with Runge-Kutta 4! 
    @property
    def name(self):
        return 'RK4'

    def update(self, t, u, f, h):
        k1 = f(t, u)
        k2 = f(t+h/2, u + h/2*k1)
        k3 = f(t+h/2, u + h/2*k2)
        k4 = f(t+h, u + h*k3)
        return u + h/6*(k1 + 2*k2 + 2*k3 + k4)

# AB2 is fundamentally different than a single-step method.
# It requires 2 points to find the next. Here, we decied that
# it will require a seed, which tells it how to take the first step.
# This is an example of "Interface Segregation".
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
