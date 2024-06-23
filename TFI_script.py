import netket as nk

import jax
from jax import numpy as jnp

import json
import pickle
import time
import pathlib

from solve import solve
from models import lind_to_pauli_strings, lind_to_pauli_strings_TFI_nd, AR_combination, RBM_dm
from ndm_ansatz import NDM, NDM_mod

###

# Example of a script used to run the algorithm on the TFI model.
# Writes the parameters of the ansatz at each timesteps in an output file

###


### Parameters of the simulation ###

gamma = 1.0
J = 1.0
h = 1.0
L = 2
lattice_dim = 1

ansatz = "NDM" # can be "AR", "RBM" or "NDM"

# NDM or RBM
alpha = 1
beta = 1
# AR
rank = 4
n_layers = 1
n_features = 1

stop_time = 10.0
dt = 0.01
method = "ptVMC" # see solve.py for the names of the other methods
order = 1

n_iters_max = 200
accuracy = 1e-9
start_learning_rate = 1e-3
Ns = 1e2
seed = 1
compute_distance = False
TFI_exact = True
mpi = False # used for parallelism
binary = False # binary=True uses json, which do not support complex numbers

output_file = "output_TFI_2spins_dt=1e-2.log"

##################

print("Building the model...")

# define the chain with L spins
g = nk.graph.Hypercube(length=L, n_dim=lattice_dim, pbc=False)

# define the Hilbert space
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)

# construct the Heisenberg hamiltonian
ha = nk.operator.LocalOperator(hi,dtype=complex) # empty hamiltonian
for j in range(L*lattice_dim): ha += -h * nk.operator.spin.sigmax(hi,j)
for (i,j) in g.edges(): # type: ignore
    ha += -J * nk.operator.spin.sigmaz(hi,i) @ nk.operator.spin.sigmaz(hi,j)

# construct the jumps operators
j_ops = []
for j in g.nodes():
    j_ops.append(jnp.sqrt(gamma) * nk.operator.spin.sigmam(hi,j))

# construct the liouvillian
lind = nk.operator.LocalLiouvillian(ha, j_ops)

# transform the liouvillian into a jax operator allowing jit compilation of its methods
if TFI_exact:
    lind_jax = lind_to_pauli_strings_TFI_nd(lind, g, h).to_jax_operator()
    lind_jax._setup() # needed to avoid a tracer error
else:
    lind_jax = lind_to_pauli_strings(lind).to_jax_operator()
    lind_jax._setup() # needed to avoid a Jax tracer error

# construct the ansatz
if ansatz == "AR":
    ansatz_ = AR_combination(rank=rank,hilbert=hi,layers=n_layers,features=n_features)
elif ansatz == "RBM":
    ansatz_ = RBM_dm(param_dtype=jnp.complex128, alpha=alpha)
elif ansatz == "NDM":
    if TFI_exact:
        ansatz_ = NDM_mod(alpha=alpha, beta=beta)
    else:
        ansatz_ = NDM(alpha=alpha, beta=beta)
else:
    raise ValueError("The ansatz can be AR, RBM or NDM")

# construct the sampler as well as the variational state
sampler = nk.sampler.MetropolisSampler( # type:ignore
                        lind.hilbert,                  # the hilbert space to be sampled
                        nk.sampler.rules.LocalRule(),  # the transition rule
                        n_chains_per_rank = 1)
vstate_var = nk.vqs.MCState(sampler, ansatz, n_samples=jnp.int64(Ns), seed=seed)

print("Solving...")

# solve the dynamics
start = time.time()
tt,pt,distt = solve(lind_jax, vstate_var, stop_time=stop_time, dt=dt,
                    method=method,
                    iters=n_iters_max, start_learning_rate=start_learning_rate, order=order, acc=accuracy,
                    compute_distance=compute_distance, mpi=mpi, TFI_exact=TFI_exact)
print(f"-----Run time: {time.time() - start} s-----")

# save the results in a file
simulation_parameters = {"model":"TFI", "ansatz":ansatz,
                         "gamma":gamma, "J":J, "h":h, "L":L, "lattice_dim":lattice_dim,
                         "alpha":alpha, "beta":beta, "rank":rank, "n_layers":n_layers, "n_features":n_features,
                         "stop_time":stop_time, "dt":dt, "method":method, "order":order,
                         "n_iters_max":n_iters_max, "accuracy":accuracy, "start_learning_rate":start_learning_rate,
                         "Ns":Ns, "seed": seed, "TFI_exact":TFI_exact}
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