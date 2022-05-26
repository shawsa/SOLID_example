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
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from time_domain import TimeDomain, TimeDomain_Start_Spacing_Stop
from time_integrator import Euler, RK4, AB2


def get_order(e0, e1, h0, h1):
    return np.log(e1/e0)/np.log(h1/h0)


def main():

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
    max_spacing = 0.1
    time = TimeDomain_Start_Spacing_Stop(t0, max_spacing, tf)
    print(f'Rectifying time step to {time.spacing} from {max_spacing}')
    # solver = Euler()
    # solver = RK4()
    # solver = AB2(Euler(), 2)
    solver = AB2(RK4(), 1)
    ts, us = solver.solve(u0, rhs, time)

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
    max_dts = [.9 * 2**(-i) for i in range(1, 6)]
    error_dict = {}
    for solver in [Euler(),
                   RK4(),
                   AB2(Euler(), 2),
                   AB2(AB2(Euler(), 2), 2)]:
        errors = []
        for max_dt in max_dts:
            time = TimeDomain_Start_Spacing_Stop(t0, max_dt, tf)
            uf = solver.t_final(u0, rhs, time)
            errors.append(uf - u_true(tf))
        error_dict[str(solver)] = errors.copy()

    # fig = plt.figure('Convergence')

    for method, errors in error_dict.items():
        order = get_order(errors[-2], errors[-1],
                          max_dts[-2], max_dts[-1])
        order_str = f' $\\mathcal{{O}}({order:.1f})$'
        plt.loglog(max_dts, errors, '.-', label=method + order_str)

    plt.xlabel(r'$\Delta_t$')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
