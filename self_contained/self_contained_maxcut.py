import networkx as nx
import numpy as np

from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective
from qokit.maxcut import maxcut_obj, get_adjacency_matrix
from qokit.utils import precompute_energies

from self_contained_utils import do_sanity_check, prepare_f

N = 5

np.random.seed(1)
G = nx.random_regular_graph(4, N, seed=42)

# For p = 1

# QOKit's function
f_qokit = get_qaoa_maxcut_objective(
    N, 1, G, parameterization="freq", objective="expectation", simulator="c"
)

# From scratch
simulator = "c"
optimization_type = "max"
precomputed_cuts = precompute_energies(maxcut_obj, N, w=get_adjacency_matrix(G))
precomputed_diagonal_hamiltonian = precomputed_cuts
precomputed_objectives = precomputed_diagonal_hamiltonian

f_ours = prepare_f(
    N,
    precomputed_objectives,
    precomputed_diagonal_hamiltonian,
    optimization_type,
    simulator,
)

# Sanity check
do_sanity_check(N, f_qokit, f_ours)
