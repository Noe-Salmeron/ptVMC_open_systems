from qutip import *
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from functools import partial

###

# Solves the dynamics exactly using qutip.

###

@partial(jax.jit, static_argnames=["model","lind"])
def to_dm(model, parameters, lind):
    # constructs the unnormalized density matrix, it can also be done directly from a variational state with .to_array().

    all_configurations = jnp.array(lind.hilbert.all_states()) # generate all configurations of the double hilbert space
    dm = jnp.exp(model.apply(parameters,all_configurations)) # vectorized density matrix
    n = 2**(len(all_configurations[0])//2)
    return jnp.reshape(dm,(n,n))



def exact_solver(lind, vstate, stop_time, dt, store_states=True, custom_tlist=None, custom_netket_e_ops=None):
    # uses qutip to solve the dynamics exactly.

    L = lind.hamiltonian.hilbert.size # number of spins
    H = lind.hamiltonian.to_qobj()
    rho0 = Qobj(np.array(to_dm(vstate.model,{"params":vstate.parameters},lind)),dims=H.dims) # initial unnormalized density matrix
    rho0 = rho0/rho0.tr()

    if custom_tlist is None:
        tlist = np.linspace(0,stop_time,jnp.int64(stop_time/dt)) # jnp.int64() is doing a floor rounding
    else:
        tlist = custom_tlist

    c_ops = [op.to_qobj() for op in lind.jump_operators] # list jump operators
    if custom_netket_e_ops is None: # compute sz for each spin by default
        e_ops = [nk.operator.spin.sigmaz(lind.hamiltonian.hilbert,i).to_qobj() for i in range(L)] # list of observables
    else:
        e_ops = [ops.to_qobj() for ops in custom_netket_e_ops]

    results = mesolve(H,rho0,tlist,c_ops,e_ops,options = Options(store_states=store_states))
    
    return results