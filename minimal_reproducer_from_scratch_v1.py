import sys
import time
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import nlopt

import qokit.parameter_utils
from qokit.qaoa_objective_labs import get_qaoa_labs_objective

def minimize_nlopt(f, x0, rhobeg=None, p=None):
    def nlopt_wrapper(x, grad):
        if grad.size > 0:
            sys.exit("Shouldn't be calling a gradient!")
        return f(x).real

    opt = nlopt.opt(nlopt.LN_BOBYQA, 2 * p)
    opt.set_min_objective(nlopt_wrapper)

    opt.set_xtol_rel(1e-8)
    opt.set_ftol_rel(1e-8)
    opt.set_initial_step(rhobeg)
    xstar = opt.optimize(x0)
    minf = opt.last_optimum_value()

    return xstar, minf

p = 30
N = 20

np.random.seed(1)

# For p = 1
f = get_qaoa_labs_objective(N, 1, parameterization="freq", objective="overlap", simulator="gpu")
best_init_u = None
best_init_v = None
best_popt = 1
tic = time.time()
for _ in range(400):
    # See paper page 5
    init_g = [np.random.uniform(0.6, 1.2 / N)]
    init_b = [np.random.uniform(0.15, 0.3)]
    init_u, init_v = qokit.parameter_utils.to_basis(init_g, init_b, basis='fourier')
    initial = np.hstack([init_u, init_v])
    x, popt = minimize_nlopt(f, initial, p=1, rhobeg=0.01 / N)
    if popt < best_popt:
        best_popt = popt
        best_init_u = x[:1]
        best_init_v = x[1:]
print("elapsed p=1", time.time() - tic)
init_u = best_init_u
init_v = best_init_v
print("best p=1", init_u, init_v)

# For the rest
tic = time.time()
for _p in range(2, p + 1):
    print(_p)
    f = get_qaoa_labs_objective(N, _p, parameterization="freq", objective="overlap", simulator="gpu")
    init_u, init_v = qokit.parameter_utils.extrapolate_parameters_in_fourier_basis(init_u, init_v, _p)
    initial = np.hstack([init_u, init_v])

    #res = scipy.optimize.minimize(
    #    f, initial, method="COBYLA", options={"rhobeg": 0.01 / N}
    #)
    x, _ = minimize_nlopt(f, initial, p=_p, rhobeg=0.01 / N)
    init_u, init_v = x[:_p],  x[_p:]
print("elapsed", time.time() - tic)


print(f"Success probability at p={p} after optimization is {1 - f(x)}")

gamma, beta = qokit.parameter_utils.from_basis(init_u, init_v, basis='fourier')
parray = range(1, p + 1)
plt.plot(parray, gamma, label=r"$\gamma$")
plt.plot(parray, beta, label=r"$\beta$")
plt.legend()
plt.savefig("figure3b.png")
