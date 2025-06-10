import time

import numpy as np
import cirq.testing as ct
from cusvaer.backends import StatevectorSimulator
from qiskit_aer import Aer
from qokit.qaoa_objective import choose_simulator
from qokit.qaoa_circuit_labs import get_parameterized_qaoa_circuit
from qokit.qaoa_objective_labs import get_precomputed_labs_merit_factors
from qokit.labs import get_energy_term_indices

N = 20
p = 30

np.random.seed(1)
gamma = np.random.random(p)
beta = np.random.random(p)

precomputed_negative_merit_factors = get_precomputed_labs_merit_factors(N)
optimization_type = "min"
precomputed_objectives = precomputed_negative_merit_factors
if optimization_type == "max":
    precomputed_objectives = -1 * np.asarray(precomputed_objectives)

def compute_objective_from_probabilities(probabilities):  # type: ignore
    if optimization_type == "max":
        return -1 * precomputed_objectives.dot(probabilities)
    return precomputed_objectives.dot(probabilities)

simulators = ["qiskit_gpu", "c", "gpu"]
svs = []
for simulator in simulators:
    print(simulator)
    terms, offset = get_energy_term_indices(N)
    if "qiskit" in simulator:
        parameterized_circuit = get_parameterized_qaoa_circuit(N, terms, p)
        qc = parameterized_circuit.assign_parameters(list(np.hstack([beta, gamma])))
        if simulator == "qiskit_cpu":
            backend = Aer.get_backend("aer_simulator_statevector")
        elif simulator == "qiskit_gpu":
            backend = StatevectorSimulator()
        else:
            raise Exception("?")
        for i in range(2):
            tic = time.time()
            sv = backend.run(qc).result().get_statevector()
            print("Elapsed sv", time.time() - tic)

        tic = time.time()
        out = np.asarray(sv)
        print("Elapsed asarray", time.time() - tic)

        tic = time.time()
        probs = np.abs(sv) ** 2
        compute_objective_from_probabilities(probs)
        print("Elapsed expectation", time.time() - tic)

    else:
        precomputed_diagonal_hamiltonian = (
            -(N**2) / (2 * precomputed_negative_merit_factors) - offset
        )
        simulator_cls = choose_simulator(name=simulator)
        sim = simulator_cls(N, terms=terms, costs=precomputed_diagonal_hamiltonian)
        n_trotters = 1
        initial_state = None
        for i in range(2):
            # Repeat so to avoid cold start
            tic = time.time()
            sv = sim.simulate_qaoa(gamma, beta, initial_state, n_trotters=n_trotters)
            print("Elapsed sv", time.time() - tic)

        tic = time.time()
        if simulator == "c":
            out = sv.get_complex()
        else:
            out = np.asarray(sv)
        print("Elapsed asarray", time.time() - tic)

        precomputed_costs = sim.get_cost_diagonal()
        tic = time.time()
        sim.get_expectation(
            sv,
            costs=precomputed_costs,
            preserve_state=False,
            optimization_type=optimization_type,
        )
        print("Elapsed expectation", time.time() - tic)

        tic = time.time()
        probs = np.abs(out) ** 2
        compute_objective_from_probabilities(probs)
        print("Elapsed expectation CPU", time.time() - tic)

    svs.append(out)

# for i in range(len(simulators) - 1):
#     ct.assert_allclose_up_to_global_phase(svs[i], svs[i + 1], atol=1e-9)
