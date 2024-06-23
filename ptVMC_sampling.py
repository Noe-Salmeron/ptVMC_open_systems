import netket as nk

import jax
import jax.numpy as jnp
import jax.flatten_util

from functools import partial
import optax
from copy import copy, deepcopy
from time import time
import collections

from models import lind_to_pauli_strings, small_evol, AR_combination, to_dm
from ptVMC_no_sampling import overlap_exact, overlap_exact_complex_grad

###

# Functions for running the normal version of the algorithm, as well as some utility or prototype functions.

###




@partial(jax.jit, static_argnums=[0,1])
def get_overlap_jit(model_var,model_fix,params_var,params_fix,x,y,cv=0.5):
    left = jnp.exp(model_fix.apply(params_fix,x))
    right = jnp.exp(model_var.apply(params_var,y))
    leftb = jnp.exp(model_var.apply(params_var,x))
    rightb = jnp.exp(model_fix.apply(params_fix,y))
    F_loc = -1 * (left / leftb) * (right / rightb)

    return jnp.real(F_loc + cv*(jnp.abs(F_loc)**2 - 1))



def estimate_cstar(vstate_var,vstate_fix):
    # estimates the optimal value for c star, and also gives the amont by which the variance will be reduced
    x = vstate_var.samples
    y = vstate_fix.samples
    N = x.shape[-1]
    x = x.reshape(-1,N)
    y = y.reshape(-1,N)

    ov_loc = jnp.real(get_overlap_jit(vstate_var.model,vstate_fix.model,{"params":vstate_var.parameters},{"params":vstate_fix.parameters},x,y,cv=0.))
    ov_loc_squared = jnp.abs(ov_loc)**2

    cov_vec = (ov_loc - jnp.mean(ov_loc)) * (ov_loc_squared - jnp.mean(ov_loc_squared))

    return -1 * jnp.mean(cov_vec) / jnp.var(ov_loc_squared), jnp.mean(cov_vec)**2 / (jnp.var(ov_loc)*jnp.var(ov_loc_squared))



@partial(jax.jit, static_argnums=[0,1,7])
def estimate_overlap_and_grad_kernel(model_var,model_fix,params_var,params_fix,x,y,cv=0.5,mpi=False):
    # jitted kernel of estimate_overlap_and_grad() below
    N = x.shape[-1]
    x = x.reshape(-1,N)
    y = y.reshape(-1,N)

    get_overlap = lambda params: get_overlap_jit(model_var,model_fix,params,params_fix,x,y,cv)

    overlap = get_overlap(params_var)

    _, f_vjp = jax.vjp(get_overlap, params_var)
    grad = f_vjp(jnp.ones_like(overlap)/overlap.size)[0]

    if not mpi:
        return (jnp.mean(overlap), jnp.var(overlap)), jax.tree_util.tree_map(jnp.conj, grad)
    else:
        from mpi4py import MPI # type:ignore
        import mpi4jax # type:ignore

        comm = MPI.COMM_WORLD
        # rank = comm.Get_rank()
        size = comm.Get_size()

        mean = jnp.mean(overlap)
        var = jnp.var(overlap)
        gradient = jax.tree_util.tree_map(jnp.conj, grad)

        mean_sum, _ = mpi4jax.allreduce(mean, op=MPI.SUM, comm=comm)
        var_sum, _ = mpi4jax.allreduce(var, op=MPI.SUM, comm=comm)
        gradient_sum = jax.tree_util.tree_map(lambda x: mpi4jax.allreduce(x, op=MPI.SUM, comm=comm)[0]/size, gradient)

        return (mean_sum/size, var_sum/size), gradient_sum



def estimate_overlap_and_grad(vstate_var,vstate_fix,cv=0.5,mpi=False):
    # computes the overlap and its gradient using vjp (grad could be used but vjp gives more control for complex numbers, grad uses vjp anyways)
    x = vstate_var.samples
    y = vstate_fix.samples
    return estimate_overlap_and_grad_kernel(vstate_var.model,vstate_fix.model,
                                            {"params":vstate_var.parameters},{"params":vstate_fix.parameters},
                                            x,y,cv,mpi=mpi)
    


def grad_distance(lind,vstate_var,vstate_fix,cv=0.5):
    # returns the sum of the distances between the elements of the estimated and exact pytree. It scales with the numbers of parameters
    # so it can't be used to compare models
    mc_grad = estimate_overlap_and_grad(vstate_var,vstate_fix,cv)[1]
    exact_grad = overlap_exact_complex_grad(lind,vstate_var.model,vstate_fix.model,{"params":vstate_var.parameters},{"params":vstate_fix.parameters})
    dist = jax.tree_util.tree_map(lambda x,y: (x-y)**2, mc_grad, exact_grad)
    return jnp.sqrt(jnp.sum(jax.flatten_util.ravel_pytree(dist)[0]))



def MP_inv(A):
    # Moore-Penrose inverse for the QGT
    eps = 1e-10
    val,vec = jnp.linalg.eig(A)
    Ainv = jnp.zeros_like(A)
    for j in range(len(val)):
        if jnp.abs(val[j]) > eps:
            Ainv += (1./val[j]) * jnp.outer(jnp.conj(vec[:,j]),vec[:,j])
    return Ainv



def Nagy_inv(A,it):
    # regularization in http://arxiv.org/pdf/1902.09483 for the QGT
    lam0 = 100
    b = 0.998
    lam_min = 1e-2
    lam = jnp.max(jnp.array([lam0*(b**it), lam_min]))
    Areg = A + lam * (jnp.identity(len(A[0]))*A)
    return jnp.linalg.inv(Areg)



from models import n_params
def apply_preconditioner(lind, vstate, grad, it=0):
    # apply a preconditioner (the inverse of the QGT) to the gradient, still experimental
    all_confs = lind.hilbert.all_states() # type:ignore
    model = vstate.model
    np = n_params(lind, model)
    S = jnp.zeros((np,np))
    parameters = {"params":vstate.parameters}
    for conf in all_confs:
        _, fun_vjp = jax.vjp(lambda x: jnp.exp(model.apply(x, conf)), parameters)
        real_part = fun_vjp(1.+0j)
        # imag_part = fun_vjp(0.-1j) # vjp takes the complex conjugate so we use -i
        # apply_grad = jax.tree_util.tree_map(lambda x,y: x+1j*y, real_part, imag_part)
        apply_grad = jax.tree_util.tree_map(jnp.conj, real_part)

        apply_grad_vec, _ = jax.flatten_util.ravel_pytree(apply_grad) # type:ignore
        S += jnp.outer(jnp.conj(apply_grad_vec), apply_grad_vec)
    
    grad_vec, unravel = jax.flatten_util.ravel_pytree(grad) # type:ignore
    new_grad_vec = Nagy_inv(S,it) @ grad_vec
    new_grad = unravel(new_grad_vec)
    return jax.tree_util.tree_map(jnp.real, new_grad)



def printmpi(str, mpi):
    # prints str on only one cpu, to avoid printing 10s of times the same message
    if mpi is True:
        from mpi4py import MPI # type:ignore

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            print(str)
    else:
        print(str)



def exact_evol_TFI(vstate_var, dt):
    # change the parameters of vstate_var to exactly apply the evolution given by the diagonal part of the liouvillian
    params = deepcopy(vstate_var.parameters)

    # hamiltonian part
    params["ZZZZrotation"] = params["ZZZZrotation"] + 0.5*dt * jnp.eye(len(params["ZZZZrotation"][0]), k=1)

    # disipative part
    params["PureSymm"]["visible_bias"] = params["PureSymm"]["visible_bias"] + 0.5*dt * jnp.ones(len(params["PureSymm"]["visible_bias"]))
    params["constant"] = params["constant"] + jnp.array([-0.5*dt * len(params["PureSymm"]["visible_bias"])]) # is technically not necessary

    vstate_var.parameters = params


def optimize(vstate_var,vstate_fix,iters,start_learning_rate,cv=0.5,acc=-1.,compute_distances=False,mpi=False):
    # optimizes the parameters of vstate_var to represent the evolved state vstate_fix
    optimizer = optax.adam(start_learning_rate)
    opt_state = optimizer.init(vstate_var.parameters)
    if vstate_fix.model.channel is not None:
        lind = vstate_fix.model.channel.lind
    else:
        lind = vstate_var.model.channel.lind

    if compute_distances:
        dist_track = [overlap_exact(lind,vstate_var.model,vstate_fix.model,
                                    {"params":vstate_var.parameters},{"params":vstate_fix.parameters})]
        var_track = [0.]
        grad_dist_track = [grad_distance(lind,vstate_var,vstate_fix,cv)]
        mean_track = [0.]
    else:
        dist_track = []
        var_track = []
        grad_dist_track = []
        mean_track = []

    overlap_track = collections.deque(maxlen=50)
    params_track = collections.deque(maxlen=50)

    for j in range(iters):
        stats, grads = estimate_overlap_and_grad(vstate_var,vstate_fix,cv,mpi=mpi)
        overlap_track.append(stats[0])
        params_track.append(deepcopy(vstate_var.parameters))

        if 1+stats[0] < acc: # stop if the desired accuracy has been reached
            printmpi(f"number of iteration steps: {j+1}", mpi)
            break
        if j == iters-1: # if the optimization step limit has been reached
            printmpi(f"number of iteration steps: {iters}, accuracy reached: {1+min(overlap_track)}", mpi)

        # grads = apply_preconditioner(lind,vstate_var,grads,it=0)
        updates, opt_state = optimizer.update(grads["params"], opt_state)
        vstate_var.parameters = optax.apply_updates(vstate_var.parameters, updates)
        
        if compute_distances:
            if j % 1 == 0:
                print(f"optimization step: {j}")
            dist_track.append(overlap_exact(lind,vstate_var.model,vstate_fix.model,
                                            {"params":vstate_var.parameters},{"params":vstate_fix.parameters}))
            var_track.append(stats[1])
            grad_dist_track.append(grad_distance(lind,vstate_var,vstate_fix,cv))
            mean_track.append(stats[0])

    _, idx = min((val, idx) for (idx, val) in enumerate(overlap_track))
    vstate_var.parameters = deepcopy(params_track[idx])

    return (dist_track, var_track, grad_dist_track, mean_track)



### Solvers ###



def ptVMC(lind, vstate_var, stop_time, dt, iters=200, start_learning_rate=1e-3, cv=0.5, acc=-1.,
          compute_distance=False, order=1, mpi=False, TFI_exact=False):
    # Euler loop by optimizing exacly at each time step, return the parameters and distance reached at each time step
    tt = [0.]
    pt = [{"params":vstate_var.parameters}]
    distt = [jnp.array(-1)]

    model_fix = vstate_var.model.__class__() # only works if classes used have default values for all of their arguments
    model_fix.__dict__ = copy(vstate_var.model.__dict__) # shallow copy because deepcopy doesn't work on hilbert spaces. Must be careful.
    model_fix.channel = small_evol(lind, dt, order=order)

    vstate_fix = nk.vqs.MCState(vstate_var.sampler, model_fix, n_samples=vstate_var.n_samples, seed=2)

    if TFI_exact:
        print("Warning: TFI_exact only works on 1D lattices")

    t = 0.
    while t < stop_time:
        printmpi(f"t={t}", mpi)
        t0 = time()

        if TFI_exact:
            exact_evol_TFI(vstate_var, dt/2) # exactly apply the diagonal part of the TFI liouvillian by changing parameters
        vstate_fix.parameters = deepcopy(vstate_var.parameters)
        optimize(vstate_var,vstate_fix,iters,start_learning_rate,cv=cv,acc=acc,mpi=mpi)
        if TFI_exact:
            exact_evol_TFI(vstate_var, dt/2) # exactly apply the diagonal part of the TFI liouvillian by changing parameters
        
        printmpi(f"optimization time : {time() - t0} s", mpi)

        if compute_distance:
            distt.append(overlap_exact(lind,vstate_var.model,vstate_fix.model,{"params":vstate_var.parameters},{"params":vstate_fix.parameters}))

        t += vstate_fix.model.channel.dt # type: ignore

        tt.append(t)
        pt.append({"params":vstate_var.parameters})

    return tt,pt,distt



def ptVMC_implicit(lind, vstate_var, stop_time, dt, iters=200, start_learning_rate=1e-3, cv=0.5, acc=-1.,
                   compute_distance=False, order=1, mpi=False, TFI_exact=False):
    # Euler loop by optimizing exacly at each time step, return the parameters and distance reached at each time step
    tt = [0.]
    pt = [{"params":vstate_var.parameters}]
    distt = [jnp.array(-1)]

    model_fix = vstate_var.model.__class__() # only works if classes used have default values for all of their arguments
    model_fix.__dict__ = copy(vstate_var.model.__dict__) # shallow copy because deepcopy doesn't work on hilbert spaces. Must be careful.
    model_fix.channel = None

    vstate_fix = nk.vqs.MCState(vstate_var.sampler, model_fix, n_samples=vstate_var.n_samples, seed=2)

    vstate_var.model.channel = small_evol(lind, -dt, order=order)

    if TFI_exact:
        print("Warning: TFI_exact only works on 1D lattices")

    t = 0.
    for _ in range(jnp.int64(stop_time/dt)):
        printmpi(f"t={t}", mpi)
        t0 = time()

        if TFI_exact:
            exact_evol_TFI(vstate_var, dt/2) # exactly apply the diagonal part of the TFI liouvillian by changing parameters
        vstate_fix.parameters = deepcopy(vstate_var.parameters)
        optimize(vstate_var,vstate_fix,iters,start_learning_rate,cv=cv,acc=acc,mpi=mpi)
        if TFI_exact:
            exact_evol_TFI(vstate_var, dt/2) # exactly apply the diagonal part of the TFI liouvillian by changing parameters
        
        printmpi(f"optimization time : {time() - t0} s", mpi)

        if compute_distance:
            distt.append(overlap_exact(lind,vstate_var.model,vstate_fix.model,{"params":vstate_var.parameters},{"params":vstate_fix.parameters}))

        t += dt

        tt.append(t)
        pt.append({"params":vstate_var.parameters})

    vstate_var.model.channel = None # undo the changes to model, useful when working with a notebook

    return tt,pt,distt



def ptVMC_factored_2nd(lind, vstate_var, stop_time, dt, iters=200, start_learning_rate=1e-3, cv=0.5, acc=-1.,
          compute_distance=False, order=1, mpi=False, TFI_exact=False):
    # Euler loop by optimizing exacly at each time step, return the parameters and distance reached at each time step
    tt = [0.]
    pt = [{"params":vstate_var.parameters}]
    distt = [jnp.array(-1)]

    model_fix = vstate_var.model.__class__() # only works if classes used have default values for all of their arguments
    model_fix.__dict__ = copy(vstate_var.model.__dict__) # shallow copy because deepcopy doesn't work on hilbert spaces. Must be careful.
    channel1 = small_evol(lind, dt, order=1, id_fact=(1+1j)/jnp.sqrt(2), l_fact=1/jnp.sqrt(2)) # type: ignore
    channel2 = small_evol(lind, dt, order=1, id_fact=(1-1j)/jnp.sqrt(2), l_fact=1/jnp.sqrt(2)) # type: ignore

    vstate_fix = nk.vqs.MCState(vstate_var.sampler, model_fix, n_samples=vstate_var.n_samples, seed=2)

    if TFI_exact:
        print("Warning: TFI_exact only works on 1D lattices")

    t = 0.
    while t < stop_time:
        printmpi(f"t={t}", mpi)
        t0 = time()

        if TFI_exact:
            exact_evol_TFI(vstate_var, dt/2) # exactly apply the diagonal part of the TFI liouvillian by changing parameters

        vstate_fix.parameters = deepcopy(vstate_var.parameters)
        vstate_fix.model.channel = channel1 # type: ignore
        optimize(vstate_var,vstate_fix,iters,start_learning_rate,cv=cv,acc=acc,mpi=mpi)

        vstate_fix.parameters = deepcopy(vstate_var.parameters)
        vstate_fix.model.channel = channel2 # type: ignore
        optimize(vstate_var,vstate_fix,iters,start_learning_rate,cv=cv,acc=acc,mpi=mpi)
        
        if TFI_exact:
            exact_evol_TFI(vstate_var, dt/2) # exactly apply the diagonal part of the TFI liouvillian by changing parameters
        
        printmpi(f"optimization time : {time() - t0} s", mpi)

        if compute_distance:
            distt.append(overlap_exact(lind,vstate_var.model,vstate_fix.model,{"params":vstate_var.parameters},{"params":vstate_fix.parameters}))

        t += vstate_fix.model.channel.dt # type: ignore

        tt.append(t)
        pt.append({"params":vstate_var.parameters})

    return tt,pt,distt



def ptVMC_factored_3rd(lind, vstate_var, stop_time, dt, iters=200, start_learning_rate=1e-3, cv=0.5, acc=-1.,
          compute_distance=False, order=1, mpi=False, TFI_exact=False):
    # Euler loop by optimizing exacly at each time step, return the parameters and distance reached at each time step
    tt = [0.]
    pt = [{"params":vstate_var.parameters}]
    distt = [jnp.array(-1)]

    model_fix = vstate_var.model.__class__() # only works if classes used have default values for all of their arguments
    model_fix.__dict__ = copy(vstate_var.model.__dict__) # shallow copy because deepcopy doesn't work on hilbert spaces. Must be careful.

    x1 = 0.87835207210750156390
    x2 = 0.38630577616990588902 - 0.99461725412242832093j
    x3 = 0.38630577616990588902 + 0.99461725412242832093j
    channel1 = small_evol(lind, dt, order=1, id_fact=x1, l_fact=6**(-1/3)) # type: ignore
    channel2 = small_evol(lind, dt, order=1, id_fact=x2, l_fact=6**(-1/3)) # type: ignore
    channel3 = small_evol(lind, dt, order=1, id_fact=x3, l_fact=6**(-1/3)) # type: ignore

    vstate_fix = nk.vqs.MCState(vstate_var.sampler, model_fix, n_samples=vstate_var.n_samples, seed=2)

    if TFI_exact:
        print("Warning: TFI_exact only works on 1D open lattices")

    t = 0.
    while t < stop_time:
        printmpi(f"t={t}", mpi)
        t0 = time()

        if TFI_exact:
            exact_evol_TFI(vstate_var, dt/2) # exactly apply the diagonal part of the TFI liouvillian by changing parameters

        vstate_fix.parameters = deepcopy(vstate_var.parameters)
        vstate_fix.model.channel = channel1 # type: ignore
        optimize(vstate_var,vstate_fix,iters,start_learning_rate,cv=cv,acc=acc,mpi=mpi)

        vstate_fix.parameters = deepcopy(vstate_var.parameters)
        vstate_fix.model.channel = channel2 # type: ignore
        optimize(vstate_var,vstate_fix,iters,start_learning_rate,cv=cv,acc=acc,mpi=mpi)

        vstate_fix.parameters = deepcopy(vstate_var.parameters)
        vstate_fix.model.channel = channel3 # type: ignore
        optimize(vstate_var,vstate_fix,iters,start_learning_rate,cv=cv,acc=acc,mpi=mpi)
        
        if TFI_exact:
            exact_evol_TFI(vstate_var, dt/2) # exactly apply the diagonal part of the TFI liouvillian by changing parameters
        
        printmpi(f"optimization time : {time() - t0} s", mpi)

        if compute_distance:
            distt.append(overlap_exact(lind,vstate_var.model,vstate_fix.model,{"params":vstate_var.parameters},{"params":vstate_fix.parameters}))

        t += vstate_fix.model.channel.dt # type: ignore

        tt.append(t)
        pt.append({"params":vstate_var.parameters})

    return tt,pt,distt