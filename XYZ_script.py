from copyreg import pickle
import netket as nk

import jax
from jax import numpy as jnp

import json
import pickle
import time
import pathlib

from solve import solve
from models import lind_to_pauli_strings, AR_combination, RBM_dm
from ndm_ansatz import NDM

###

# Example of a script to run the algorithm on the XYZ model.

###


### Parameters of the simulation ###

gamma = 1.0
Jx = 0.9
Jy = 0.95
Jz = 1.0
L = 4
lattice_dim = 1

ansatz = "NDM" # can be "AR", "RBM" or "NDM"

# NDM or RBM
alpha = 2
beta = 2
# AR
rank = 4
n_layers = 1
n_features = 1


stop_time = 10.0
dt = 0.01
method = "ptVMC" # see solve.py for the names of the other methods
order = 1

n_iters_max = 150
accuracy = -1.0
start_learning_rate = 1e-3
Ns = 1e2
seed = 1
compute_distance = False
mpi = False # for parallelisation
binary = False # binary=True uses Json, which does not support complex numbers

output_file = "output_4spins_dt=1e-2_Ns=1e2.log"

##################

print("Building the model...")

# define the chain with L spins
g = nk.graph.Hypercube(length=L, n_dim=lattice_dim, pbc=False)

# define the Hilbert space
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)

# construct the Heisenberg hamiltonian
ha = nk.operator.LocalOperator(hi,dtype=complex) # empty hamiltonian
for (i,j) in g.edges(): # type: ignore
    ha += Jx * nk.operator.spin.sigmax(hi,i) @ nk.operator.spin.sigmax(hi,j) +\
        Jy * nk.operator.spin.sigmay(hi,i) @ nk.operator.spin.sigmay(hi,j) +\
        Jz * nk.operator.spin.sigmaz(hi,i) @ nk.operator.spin.sigmaz(hi,j)

# construct the jumps operators
j_ops = []
for j in g.nodes():
    j_ops.append(jnp.sqrt(gamma) * nk.operator.spin.sigmam(hi,j))

# construct the liouvillian
lind = nk.operator.LocalLiouvillian(ha, j_ops)

# transform the liouvillian into a jax operator allowing jit compilation of its methods
lind_jax = lind_to_pauli_strings(lind).to_jax_operator()
lind_jax._setup() # needed to avoid a Jax tracer error

# some dense operators to compute expectation values later
# magnetization = sum([nk.operator.spin.sigmaz(hi,i) for i in range(L)])/L
# sz = [jnp.array(nk.operator.spin.sigmaz(hi,i).to_dense()) for i in range(L)]

# construct the ansatz
if ansatz == "AR":
    ansatz_ = AR_combination(rank=rank,hilbert=hi,layers=n_layers,features=n_features)
elif ansatz == "RBM":
    ansatz_ = RBM_dm(param_dtype=jnp.complex128, alpha=alpha)
elif ansatz == "NDM":
    ansatz_ = NDM(alpha=alpha, beta=beta)
else:
    raise ValueError("The ansatz can be AR, RBM or NDM")

# construct the sampler as well as the variational state
sampler = nk.sampler.MetropolisSampler( # type:ignore
                        lind.hilbert,                  # the hilbert space to be sampled
                        nk.sampler.rules.LocalRule(),  # the transition rule
                        n_chains_per_rank = 1)
vstate_var = nk.vqs.MCState(sampler, ansatz_, n_samples=jnp.int64(Ns), seed=seed)

print("Solving...")

# solve the dynamics
start = time.time()
tt,pt,distt = solve(lind_jax, vstate_var, stop_time=stop_time, dt=dt,
                    method=method,
                    iters=n_iters_max, start_learning_rate=start_learning_rate, order=order, acc=accuracy,
                    compute_distance=compute_distance, mpi=mpi)
print(f"-----Run time: {time.time() - start} s-----")

# save the results in a file
simulation_parameters = {"model":"Heisenberg", "ansatz":ansatz,
                         "gamma":gamma, "Jx":Jx, "Jy":Jy, "Jz":Jz, "L":L, "lattice_dim":lattice_dim,
                         "alpha":alpha, "beta":beta, "rank":rank, "n_layers":n_layers, "n_features":n_features,
                         "stop_time":stop_time, "dt":dt, "method":method, "order":order,
                         "n_iters_max":n_iters_max, "accuracy":accuracy, "start_learning_rate":start_learning_rate,
                         "Ns":Ns, "seed": seed}
if not binary: 
    pt_list = jax.tree_util.tree_map(lambda x: x.tolist(), pt)
else:
    pt_list = pt
results = {"times":tt, "parameters":pt_list, 
           "distances":jax.tree_util.tree_map(lambda x: x.tolist(), distt), 
           "simulation_parameters":simulation_parameters}

fn = pathlib.Path(__file__).parent / output_file
if not binary:
    with open(fn, "w") as f:
        json.dump(results, f)
else:
    with open(fn, "wb") as f:
        pickle.dump(results, f)