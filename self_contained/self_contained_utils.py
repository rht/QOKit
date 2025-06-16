import numpy as np
import qokit.parameter_utils
from qokit.qaoa_objective import choose_simulator
import scipy.optimize


def prepare_f(
    N,
    precomputed_objectives,
    precomputed_diagonal_hamiltonian,
    optimization_type,
    simulator,
):
    simulator_cls = choose_simulator(name=simulator)
    sim = simulator_cls(N, costs=precomputed_diagonal_hamiltonian)

    def compute_objective_from_probabilities(probabilities):  # type: ignore
        if optimization_type == "max":
            return -1 * precomputed_objectives.dot(probabilities)
        return precomputed_objectives.dot(probabilities)

    def f_ours(*args, return_probs=False):
        n_trotters = 1
        initial_state = None
        parameterization = "freq"
        gamma, beta = qokit.parameter_utils.convert_to_gamma_beta(
            *args, parameterization=parameterization
        )
        sv = sim.simulate_qaoa(gamma, beta, initial_state, n_trotters=n_trotters)
        if simulator == "c":
            sv = sv.get_complex()
        else:
            sv = np.asarray(sv)
        probs = np.abs(sv) ** 2
        if return_probs:
            return probs
        return compute_objective_from_probabilities(probs)

    return f_ours


def do_sanity_check(N, f_qokit, f_ours):
    init_g = [np.random.uniform(0.6, 1.2 / N)]
    init_b = [np.random.uniform(0.15, 0.3)]
    init_u, init_v = qokit.parameter_utils.to_basis(init_g, init_b, basis="fourier")
    initial = np.hstack([init_u, init_v])
    for f in [f_qokit, f_ours]:
        res = scipy.optimize.minimize(
            f, initial, method="COBYLA", options={"rhobeg": 0.01 / N}
        )
        print(res)
