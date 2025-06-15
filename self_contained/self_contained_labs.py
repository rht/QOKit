from pathlib import Path
import sys

import numpy as np

from qokit.qaoa_objective_labs import get_qaoa_labs_objective
from qokit.qaoa_objective_labs import get_precomputed_labs_merit_factors
from qokit.labs import get_energy_term_indices

sys.path.append(str(Path(__file__).parent))
from self_contained_utils import do_sanity_check, prepare_f

N = 5

np.random.seed(1)

# For p = 1

# QOKit's function
f_qokit = get_qaoa_labs_objective(
    N, 1, parameterization="freq", objective="expectation", simulator="c"
)

# From scratch
simulator = "c"
precomputed_negative_merit_factors = get_precomputed_labs_merit_factors(N)
"""get_energy_term_indices
Return indices of Pauli Zs in the LABS problem definition

Parameters
----------
N : int
    Problem size (number of spins)

Returns
-------
terms : list of tuples
    List of tuples, where each tuple defines a summand
    and contains indices of the Pauli Zs in the product
    e.g. if terms = [(0,1), (0,1,2,3), (1,2)]
    the Hamiltonian is Z0Z1 + Z0Z1Z2Z3 + Z1Z2

offset : int
    energy offset required due to constant factors (identity terms)
    not included in the Hamiltonian
"""
_, offset = get_energy_term_indices(N)
precomputed_diagonal_hamiltonian = (
    -(N**2) / (2 * precomputed_negative_merit_factors) - offset
)
optimization_type = "min"
precomputed_objectives = precomputed_negative_merit_factors


f_ours = prepare_f(
    N,
    precomputed_objectives,
    precomputed_diagonal_hamiltonian,
    optimization_type,
    simulator,
)

# Sanity check
do_sanity_check(N, f_qokit, f_ours)
