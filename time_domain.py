'''A time-discretization for an ODE solver.'''

# The old code required the start time and exactly two of the three 
# time parameters: stop, steps, spacing. Depending on which were provided
# it would run some code and end with a start time, a spacing, and a
# number of steps. Using if-blocks and asserts to enforce this is
# an anit-pattern. There is a reason Python doesn't have switch 
# statements!

# We can employ the "Strategy Pattern" here. When implemented in OO,
# each strategy is a class (unit) that does a single thing. We ensure
# That each strategy (class) obeys the Listov Substitution princle in
# Solid. Then instead of passing two out of the three parameters,
# We select the class that has the init function signature we want to use.

import math
import numpy as np

class TimeDomain:
    
    def __init__(self, start, spacing, steps):
        self.start = start
        self.spacing = spacing
        self.steps = steps
        self.initialze_array()

    def initialze_array(self):
        self.array = self.start + self.spacing*np.arange(self.steps+1)

class TimeDomain_Start_Stop_MaxSpacing(TimeDomain):

    # Notice here that the variable is called max_spacing, not spacing. The old 
    # code simply called it "dt" and if it needed a different spacing, it would
    # change it, and print a line letting us know. This is bad. The variable
    # name meant different things depending on the strategy chosen. The code
    # hid this (some would say it lied!) and the author felt the need to 
    # highlilght it with a print statement.
    def __init__(self, start, stop, max_spacing):
        self.start = start
        self.steps = math.ceil((stop-start)/max_spacing)
        self.spacing = (stop - start)/self.steps
        self.initialze_array()


class TimeDomain_Start_Stop_Steps(TimeDomain):
    def __init__(self, start, stop, steps):
        self.start = start
        self.steps = steps
        self.spacing = (stop - start)/steps
        self.initialze_array()

