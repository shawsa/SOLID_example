'''
Specifications of the assignment:
    Solves a time dependent ODE specified by
        a rhs forcing term (eg y' = f(t, y))
        and an initial value y(0) = y_0
    Can optionally use any of the following time-integration methods (link):
        Forward Euler
        Adams-Bashforth 2
        (optional) RK4
    Can provide the solution in the following ways
        The value of the solution at a given point in time y(t) = ...
        The value of the solution at equally spaced points for
            with a specific number of steps and step-size
            with a specific number of steps and an end time
            with a maximum step-size and an end time
    Demonstrate correctness with the following
        Plot the exact and approximate solutions (error on semi-log)
'''
import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym


# These function are the essence of forward-Euler, RK4, and single-step
# methods in general. They encapsulate the behaviour, but they can't
# be easily used on their own without the looping functionality. They
# have the same function signature. It serves as an interface, and it
# is the essence of a single-step method.
def euler_update(t, u, f, h):
    return u + h*f(t, u)


def RK4_update(t, u, f, h):
    k1 = f(t, u)
    k2 = f(t+h/2, u + h/2*k1)
    k3 = f(t+h/2, u + h/2*k2)
    k4 = f(t+h, u + h*k3)
    return u + h/6*(k1 + 2*k2 + 2*k3 + k4)

# Notice that AB2 has a different function signature than the single-step methods.
# We will want to segregate these interfaces.

def AB2_update(t, y, y_old, f, h):
    return y + 3/2*h*f(t, y) - 1/2*h*f(t-h, y_old)


# The bulk of our refactoring will be in this function. It violates almost all
# of the solid principles.
def solve(t0, u0, rhs, *,
          dt=None, steps=None, tf=None,
          ret='last', method='euler'):
    # The function signature shows that it does more than one thing. In fact,
    # it does six things. There are three methods and 2 kinds of return-value
    # behavior.
    assert method in ['euler', 'RK4', 'AB2'], \
        f'Error: method  "{method}" not recognized.'
    assert ret in ['last', 'all'], \
        f'Error: Return parameter  "{ret}" not recognized.'

    if tf is None:
        assert steps is not None
        assert dt is not None
    elif steps is None:
        assert dt is not None
        assert tf is not None
        steps = math.ceil((tf-t0)/dt)
        new_dt = (tf - t0)/steps
        # This print statement is bad. The variable "dt" does not
        # mean the same thign in this case. The variable name is a 
        # lie, and the author felt guilty about it, so they added
        # this print statement. See the time_domain.py file for more.
        print(f'Rectifying time step to {new_dt:.5g} from {dt:.5g}.')
        dt = new_dt
    elif dt is None:
        assert tf is not None
        assert steps is not None
        dt = (tf - t0)/steps
    else:
        raise ValueError('Specify only two of "dt", "steps", or "tf".')


    ts = t0 + dt*np.arange(steps+1)
    if ret == 'all':
        us = [u0]
    u = u0

    # This design required nested if statements to get the six behaviours.
    # If we add a new return value, we would have to add that functionality
    # in three places. We could have nested the method-if inside the return-if
    # but then we would need to repeat each method when we add a new return value.
    # The diversity of methods and return values are coupled here. We need to make a 
    # descision and it makes adding one kind of feature more difficult than another.
    # This violates open/closed. If we add a new method, we need to change this 
    # block and re-implement the return-if feature for the new method.
    # We can avoid this with dependency injection.
    if method == 'euler':
        for t in ts[:-1]:
            u = euler_update(t, u, rhs, dt)
            if ret == 'all':
                us.append(u)
    elif method == 'RK4':
        for t in ts[:-1]:
            u = RK4_update(t, u, rhs, dt)
            if ret == 'all':
                us.append(u)
    elif method == 'AB2':
        # Notice that here, the Euler code looping is kind of repeated! 
        # Also, the author gave up on trying to programatically change the
        # behavior and in order to use a different seed, we need to 
        # comment/uncomment certain code. This violates single-responsibility
        # and open/closed.

        # seed with forward euler
        u_old = u
        # u = euler_update(t0, u, rhs, dt/2)
        # u = euler_update(t0+dt/2, u, rhs, dt/2)
        # seed with RK4
        u = RK4_update(t0, u, rhs, dt)
        if ret == 'all':
            us.append(u)
        for t in ts[1:-1]:
            u, u_old = AB2_update(t, u, u_old, rhs, dt), u
            if ret == 'all':
                us.append(u)
    else:
        raise ValueError('The program should never reach this point.')
        # This is redundant with the asserts above. If we added a value to the
        # assert list above but didn't add the option to this switch (say, because we
        # had a spelling error) then this function would still return, without throwing
        # any errors. This redundtant line serves to highlight the coupling between
        # this switch block and the asserts above. This is not ideal.

    if ret == 'all':
        return ts, us
    elif ret == 'last':
        return u
    else:
        raise ValueError('The program should never reach this point.')

def get_order(e0, e1, h0, h1):
    return np.log(e1/e0)/np.log(h1/h0)

# This is the start of the driver script. It should be nested in an if __name__ ==
# block so that the functions can be reused without executing the plotting.
# This is just a nice python-specific thing that isn't specifically part of SOILD.

###################
# manufactured solution
###################
t_sym = sym.symbols('t')
u_sym = sym.symbols('u')
u_true_sym = sym.sin(t_sym)
u_true = sym.lambdify(t_sym, u_true_sym)
rhs_sym = u_true_sym.diff('t')
rhs_lambdified = sym.lambdify(t_sym, rhs_sym)
def rhs(t, u):
    return rhs_lambdified(t)

t0 = 0
u0 = u_true(t0)

###################
# Solution over time
###################

tf = 10
dt = 0.1

# A forward euler time-integrator isn't a string. It has a name, but
# using a string to change the effect of a function call violates the
# single-responsibility rule.

method = 'euler'

# Notice that with ret='all' this returns a pair of arrays.
ts, us = solve(t0, u0, rhs, dt=dt, tf=tf, ret='all', method=method)

fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

axes[0].plot(ts, u_true(ts), 'b-', label='exact')
axes[0].plot(ts, us, 'g-', label='approximate')
axes[0].legend()
axes[0].set_xlabel('$t$')

axes[1].semilogy(ts, (us - u_true(ts)), 'k-', label='error')
axes[1].legend()
plt.suptitle('Solution over time.')
plt.show()

###################
# Convergence
###################

tf = 1_000
dts = [.9 * 2**(-i) for i in range(1, 6)]
error_dict = {}
for method in ['euler', 'RK4', 'AB2']:
    errors = []
    for dt in dts:
        # Notice that with ret='last' this returns a solution at the final time (float).
        uf = solve(t0, u0, rhs, dt=dt, tf=tf, ret='last', method=method)
        errors.append(uf - u_true(tf))
    error_dict[method] = errors.copy()

# fig = plt.figure('Convergence')

for method, errors in error_dict.items():
    order = get_order(errors[-2], errors[-1], dts[-2], dts[-1])
    order_str = f' $\\mathcal{{O}}({order:.1f})$'
    # In the legend, I'm relying on the method being a string.
    # This creates a hidden dependency. If I change what method is,
    # then this part of the code will change as well.
    plt.loglog(dts, errors, '.-', label=method + order_str)

plt.xlabel(r'$\Delta_t$')
plt.ylabel('Error')
plt.legend()
plt.show()
