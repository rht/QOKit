# 0)  Imports ─ still nothing but NetworkX, Qiskit-core and SciPy
import networkx as nx, numpy as np
from scipy.optimize import minimize
from qiskit_aer.primitives import Estimator
from qiskit_aer.primitives import Sampler
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# 1)  Balanced binary tree of height-2  (unique MIS = {0,3,4,5,6})
g = nx.balanced_tree(r=2, h=2)
target_size = 5  # we KNOW this graph’s MIS size

g = nx.Graph()
g.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (2, 7)])
target_size = 6
n = g.number_of_nodes()


# 2)  MIS → Ising  (same derivation as before)
def mis_coeffs(graph, A=2.0):
    h = [0.5 - (A / 4) * graph.degree(v) for v in graph]
    J = {(u, v): A / 4 for u, v in graph.edges()}
    return h, J


def pauli(indices):
    label = ["I"] * n
    for i in indices:
        label[n - 1 - i] = "Z"
    return "".join(label)


def cost_op(graph, A=2.0):
    h, J = mis_coeffs(graph, A)
    labels, coeffs = [], []
    for i, h_i in enumerate(h):  # linear Z terms
        labels.append(pauli([i]))
        coeffs.append(h_i)
    for (i, j), J_ij in J.items():  # ZZ terms
        labels.append(pauli([i, j]))
        coeffs.append(J_ij)
    return SparsePauliOp(labels, coeffs)


# 3)  QAOA ansatz factory
def qaoa_ansatz(p, h, J):
    γ = ParameterVector("γ", p)
    β = ParameterVector("β", p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(p):
        # cost-unitary
        for (i, j), Jij in J.items():
            qc.cx(i, j)
            qc.rz(2 * Jij * γ[layer], j)
            qc.cx(i, j)
        for i, h_i in enumerate(h):
            qc.rz(2 * h_i * γ[layer], i)
        # mixer
        qc.rx(2 * β[layer], range(n))
    qc.measure_all()
    return qc, γ, β


# 4)  Build once
A = 2.0
h_vec, J_dict = mis_coeffs(g, A)
C_op = cost_op(g, A)
p_layers = 4
ansatz, γ, β = qaoa_ansatz(p_layers, h_vec, J_dict)
estimator = Estimator()
sampler = Sampler()


# utility → bind regardless of SDK version
def bind(circ, m):
    return (
        circ.bind_parameters(m)
        if hasattr(circ, "bind_parameters")
        else circ.assign_parameters(m, inplace=False)
    )


# 5)  Loop over random restarts until we hit MIS-size 5
attempt = 0
while True:
    attempt += 1
    seed = 1234 + attempt
    rng = np.random.default_rng(seed)
    x0 = 0.1 * rng.standard_normal(2 * p_layers)

    def energy(params):
        b = {γ[i]: params[i] for i in range(p_layers)}
        b.update({β[i]: params[p_layers + i] for i in range(p_layers)})
        return estimator.run(bind(ansatz, b), C_op).result().values[0]

    res = minimize(energy, x0, method="COBYLA", options={"maxiter": 800, "rhobeg": 0.5})
    # sample once at the optimum
    bopt = {γ[i]: res.x[i] for i in range(p_layers)}
    bopt.update({β[i]: res.x[p_layers + i] for i in range(p_layers)})
    quasi = sampler.run(bind(ansatz, bopt), shots=1024).result().quasi_dists[0]
    bit_int = max(quasi, key=quasi.get)
    bitstr = format(bit_int, f"0{n}b")
    ind_set = [i for i, b in enumerate(bitstr[::-1]) if b == "1"]

    if len(ind_set) == target_size:
        print(f"\nSUCCESS after {attempt} restart(s)")
        print("bit-string :", bitstr)
        print("MIS       :", ind_set, f"(size = {target_size})")
        break
    else:
        print(f"∘ restart {attempt:2d}: got {ind_set}  (size {len(ind_set)})")
