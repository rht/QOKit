import time

import numpy as np
import cirq.testing as ct
from qiskit_aer import Aer
from qokit.qaoa_objective import choose_simulator
from qokit.qaoa_circuit_labs import get_parameterized_qaoa_circuit
from qokit.qaoa_objective_labs import get_precomputed_labs_merit_factors
from qokit.labs import get_energy_term_indices

N = 5
p = 10

np.random.seed(1)
gamma = np.random.random(p)
beta = np.random.random(p)

precomputed_negative_merit_factors = get_precomputed_labs_merit_factors(N)

simulators = ["qiskit_gpu", "c", "gpu"]
simulators = ["qiskit_cpu", "c"]
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
            from cusvaer.backends import StatevectorSimulator

            backend = StatevectorSimulator()
        else:
            raise Exception("?")
        tic = time.time()
        sv = np.asarray(backend.run(qc).result().get_statevector())
        print("Elapsed sv", time.time() - tic)

    else:
        precomputed_diagonal_hamiltonian = (
            -(N**2) / (2 * precomputed_negative_merit_factors) - offset
        )
        simulator_cls = choose_simulator(name=simulator)
        sim = simulator_cls(N, terms=terms, costs=precomputed_diagonal_hamiltonian)
        n_trotters = 1
        initial_state = None
        tic = time.time()
        sv = sim.simulate_qaoa(gamma, beta, initial_state, n_trotters=n_trotters)
        print("Elapsed sv", time.time() - tic)
        if simulator == "c":
            sv = sv.get_complex()
        else:
            sv = np.asarray(sv)

    svs.append(sv)

for i in range(len(simulators) - 1):
    ct.assert_allclose_up_to_global_phase(svs[i], svs[i + 1], atol=1e-9)
