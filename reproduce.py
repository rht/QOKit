from pathlib import Path
import time

import numpy as np
from qiskit_aer import Aer
from qiskit_aer.primitives import SamplerV2 as Sampler
import scipy.optimize

import qokit
from qokit.utils import brute_force
from qokit.utils import precompute_energies
import qokit.parameter_utils
from qokit.labs import (
    get_energy_term_indices,
    negative_merit_factor_from_bitstring,
    merit_factor as merit_factor_fn,
)
from qokit.qaoa_circuit_labs import get_parameterized_qaoa_circuit

# For comparison
from qokit.qaoa_objective_labs import get_qaoa_labs_objective as full_version

np.random.seed(1)


def get_qaoa_objective(
    N: int,
    precomputed_objectives=None,
    terms=None,
    parameterization="theta",
    objective: str = "expectation",
    parameterized_circuit=None,
    mixer: str = "x",
    initial_state: np.ndarray | None = None,
    n_trotters: int = 1,
    optimization_type="min",
):
    assert objective == "overlap"

    if optimization_type == "max":
        precomputed_objectives = -1 * np.asarray(precomputed_objectives)
    minval = precomputed_objectives.min()
    bitstring_loc = (precomputed_objectives == minval).nonzero()
    assert len(bitstring_loc) == 1
    bitstring_loc = bitstring_loc[0]

    def compute_objective_from_probabilities(probabilities):  # type: ignore
        # compute overlap
        overlap = 0
        for i in range(len(bitstring_loc)):
            overlap += probabilities[bitstring_loc[i]]
        return 1 - overlap

    assert mixer == "x"
    backend = Aer.get_backend("aer_simulator_statevector")

    def g(gamma, beta):
        qc = parameterized_circuit.assign_parameters(list(np.hstack([beta, gamma])))
        sv = np.asarray(backend.run(qc).result().get_statevector())
        probs = np.abs(sv) ** 2
        return compute_objective_from_probabilities(probs)

    def fq(*args):
        gamma, beta = qokit.parameter_utils.convert_to_gamma_beta(
            *args, parameterization=parameterization
        )
        return g(gamma, beta)

    return fq


def get_precomputed_merit_factors(N: int):
    prefix = Path(qokit.__file__).parents[1] / "examples"
    fpath = Path(
        prefix,
        f"assets/precomputed_merit_factors/precomputed_energies_{N}.npy",
    )
    if N > 10 and fpath.exists():
        # load from disk
        ens = np.load(fpath)
    else:
        # precompute
        #if N > 10 and N <= 24:
        #    raise RuntimeError(
        #        f"""
#Failed to load from {fpath}, attempting to recompute for N={N},
#Precomputed energies should be loaded from disk instead. Run assets/load_assets_from_s3.sh to obtain precomputed energies
        #        """
        #    )
        ens = precompute_energies(negative_merit_factor_from_bitstring, N)
        np.save(fpath, ens)
    return ens


def get_qaoa_labs_objective(
    N: int,
    p: int,
):
    precomputed_negative_merit_factors = get_precomputed_merit_factors(N)

    assert p is not None, "p must be passed if simulator == 'qiskit'"
    terms, _ = get_energy_term_indices(N)
    parameterized_circuit = get_parameterized_qaoa_circuit(N, terms, p)
    parameterization = "theta"
    objective = "overlap"

    return get_qaoa_objective(
        N=N,
        precomputed_objectives=precomputed_negative_merit_factors,
        parameterized_circuit=parameterized_circuit,
        parameterization=parameterization,
        objective=objective,
    )


p = 3
gamma = np.random.random(p)
beta = np.random.random(p)
N = 5
initial = np.hstack([gamma, beta])
f = get_qaoa_labs_objective(N, p)
if 1:
    print(f"Success probability at p={p} before optimization is {1 - f(initial)}")

    f2 = full_version(N, p, parameterization="theta", objective="overlap")
    print(f"Success probability at p={p} before optimization is {1 - f2(initial)}")

    tic = time.time()
    res = scipy.optimize.minimize(
        f, initial, method="COBYLA", options={"rhobeg": 0.01 / N}
    )
    print("Optimization elapsed:", time.time() - tic)
    print(f"Success probability at p={p} after optimization is {1 - f2(res.x)}")
    print(list(res.x))
    optimized = res.x


# Solving it
terms, _ = get_energy_term_indices(N)
parameterized_circuit = get_parameterized_qaoa_circuit(N, terms, p)
qc = parameterized_circuit.assign_parameters(optimized)
qc.measure_all()

sampler = Sampler()  # You can pass backend / noise model here
job = sampler.run([qc], shots=1000)  # shots=None would give exact probabilities
res = job.result()[0]

bitstrings = res.data.meas.get_bitstrings()
bitstrings = [np.array([1 if e == "1" else -1 for e in b]) for b in bitstrings]
# bitstrings  = ["".join(map(str, b[::-1])) for b in bitstrings]
best_mf = 0
best_bs = ""

for bs in bitstrings:
    mf = merit_factor_fn(bs, N)
    if mf > best_mf:
        best_mf = mf
        best_bs = bs
print("best qaoa", best_mf, [1 if e == 1 else 0 for e in best_bs])
print("brute force", brute_force(merit_factor_fn, N))
