import networkx as nx
import numpy as np
import scipy.optimize

import qokit.parameter_utils
from qokit.fur.diagonal_precomputation import precompute_vectorized_cpu_parallel

from self_contained_utils import do_sanity_check, prepare_f

np.random.seed(1)
g = nx.balanced_tree(r=2, h=2)

g = nx.Graph()
g.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (2, 7)])
N = g.number_of_nodes()

# ──────────────────────────────────────────────────────────────
# 2.  Encode MIS → Ising   C = Σ hᵢ Zᵢ + Σ Jᵢⱼ ZᵢZⱼ
#     QOKit expects a flat list  [(coeff, (i,)), (coeff, (i,j)), …]
# ──────────────────────────────────────────────────────────────
A_penalty = 2.0  # ≥1 so penalties dominate

terms = []  # (coeff, qubit-tuple)

# single-qubit Z terms
for v in g.nodes():
    coeff = 0.5 - (A_penalty / 4) * g.degree(v)
    terms.append((coeff, (v,)))

# ZZ terms for every edge
for u, v in g.edges():
    coeff = A_penalty / 4
    terms.append((coeff, (u, v)))


# From scratch
simulator = "c"
optimization_type = "min"
precomputed_objectives = precomputed_diagonal_hamiltonian = (
    precompute_vectorized_cpu_parallel(terms, 0.0, N)
)

f_ours = prepare_f(
    N,
    precomputed_objectives,
    precomputed_diagonal_hamiltonian,
    optimization_type,
    simulator,
)

# Sanity check
p = 4
init_g = [np.random.uniform(0.6, 1.2 / N) for _ in range(p)]
init_b = [np.random.uniform(0.15, 0.3) for _ in range(p)]
init_u, init_v = qokit.parameter_utils.to_basis(init_g, init_b, basis="fourier")
initial = np.hstack([init_u, init_v])
res = scipy.optimize.minimize(
    f_ours, initial, method="COBYLA", options={"rhobeg": 0.01 / N}
)
print(res.x)
probs = f_ours(res.x, return_probs=True)
best = np.argmax(probs)
bitstr = format(best, f"0{N}b")  # e.g. "1111100"
indset = [i for i, b in enumerate(bitstr[::-1]) if b == "1"]

print("bit-string :", bitstr)
print("independent set:", indset, "(size =", len(indset), ")")
