import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt

# ▸ Qiskit imports (2025.x API)
from qiskit.primitives import Sampler          # noise-aware primitive
from qiskit_algorithms import QAOA             # variational quantum algo
from qiskit_algorithms.optimizers import COBYLA  # gradient-free optimiser
from qiskit_optimization.applications import StableSet
from qiskit_optimization.algorithms import MinimumEigenOptimizer

np.random.seed(1)
# g = nx.cycle_graph(5)  # degenerate solutions

# g = nx.star_graph(4) 
g = nx.balanced_tree(r=2, h=2)

g = nx.Graph()
g.add_edges_from([(0,1),(0,2),(1,3),(1,4),(2,5),(2,6),(2,7)])

#nx.draw_circular(g, with_labels=True, node_color="#F7CE5B")
#plt.show()

# 2) Turn the MIS instance into a QuadraticProgram
mis = StableSet(g)              # ↔ maximum-independent-set application
qp  = mis.to_quadratic_program()

# 3) Configure QAOA
sampler   = Sampler()           # local Aer simulator by default
optimizer = COBYLA(maxiter=250) # purely derivative-free search
qaoa      = QAOA(sampler=sampler, optimizer=optimizer, reps=2)

# 4) Wrap in a high-level optimizer
meo = MinimumEigenOptimizer(qaoa)

# 5) Solve and analyse
result    = meo.solve(qp)
iset      = mis.interpret(result)

print("Optimal independent-set:", iset)
print("Set size (objective):", result.fval)
