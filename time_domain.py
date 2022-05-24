
'''A time domain class for use in an ODE solver.'''
import math
import numpy as np


class TimeDomain:
    def __init__(self, *, start, steps, spacing):
        self.start = start
        self.steps = steps
        self.spacing = spacing

        self.initialze_array()

    def initialze_array(self):
        self.array = self.start + self.spacing*np.arange(self.steps+1)



class TimeDomain_Start_Stop_Steps(TimeDomain):
    def __init__(self, *, start, end, steps):
        self.start = start
        self.steps = steps
        self.spacing = (end - start)/(steps)
        self.initialze_array()

class TimeDomain_Start_Stop_MaxSpacing(TimeDomain):

    def __init__(self, *, start, end, max_spacing):
        self.start = start
        self.steps = math.ceil((end - start)/max_spacing)
        self.spacing = (end - start)/(self.steps)
        self.initialze_array()
