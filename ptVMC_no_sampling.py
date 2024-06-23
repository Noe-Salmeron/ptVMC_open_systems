import jax
import jax.numpy as jnp

from functools import partial
import optax
from copy import copy, deepcopy

from models import small_evol, AR_combination

###

# Functions to run the algorithm without sampling.

###



@partial(jax.jit, static_argnames=["model"])
def to_dm(model,parameters,lind):
    # returns the non normalized density matrix
    all_configurations = jnp.array(lind.hilbert.all_states())
    dm = jnp.exp(model.apply(parameters,all_configurations))
    n = 2**(len(all_configurations[0])//2)
    return jnp.reshape(dm,(n,n))



def overlap_exact(lind,model1,model2,parameters1,parameters2):
    # compute the distance between the state with parameters1 and the state with parameters2 evolved exactly by dt
    dm1 = to_dm(model1,parameters1,lind)
    dm2 = to_dm(model2,parameters2,lind)

    dag = lambda x: jnp.conj(jnp.transpose(x))
    return jnp.real(-1*jnp.abs(jnp.trace(dag(dm1) @ dm2))**2 / (jnp.trace(dag(dm1)@dm1) * jnp.trace(dag(dm2)@dm2)))



@partial(jax.jit, static_argnames=["model1","model2"])
def overlap_exact_complex_grad(lind,model1,model2,parameters1,parameters2):
    # take the gradient of the exact distance with vjp
    f = lambda param: overlap_exact(lind,model1,model2,param,parameters2)
    _, f_vjp = jax.vjp(f,parameters1)
    return jax.tree_util.tree_map(jnp.conj, f_vjp(1.0)[0])



@partial(jax.jit, static_argnames=["model1","model2","iters","compute_distances"])
def optimize_exact(lind,model1,model2,parameters,iters,start_learning_rate,compute_distances=False):
    # optimize parameters to represent the exact evolved state during a time dt. Uses adam and a weird for loop to make it efficiently jittable.
    optimizer = optax.adam(start_learning_rate)

    def body_fun_all(lind,model1,model2,params,parameters,opt_state,dist_track,i):
        dist_track = dist_track.at[i].set(overlap_exact(lind,model1,model2,params,parameters))
        grads = overlap_exact_complex_grad(lind,model1,model2,params,parameters)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, dist_track)

    body_fun = lambda i, carry: body_fun_all(lind,model1,model2,carry[0],parameters,carry[1],carry[2],i)

    return jax.lax.fori_loop(lower=0,upper=iters,body_fun=body_fun,
                             init_val=(deepcopy(parameters),optimizer.init(deepcopy(parameters)),jnp.zeros(iters)))




### Solvers ###


def ptVMC_exact(lind,model,parameters,stop_time,dt,iters=1000,start_learning_rate=1e-3,order=1,**kwargs):
    # Euler loop by optimizing exacly at each time step, return the parameters and distance reached at each time step

    model2 = model.__class__() # only works if classes used have default values for all their arguments
    model2.__dict__ = copy(model.__dict__) # shallow copy because deepcopy doesn't work on hilbert spaces. Must be careful
    model2.channel = small_evol(lind, dt, order=order)

    tt = [0.]
    pt = [parameters]
    distt = [jnp.array(-1)]

    t = 0.
    for _ in range(int(stop_time/dt)):
        params = optimize_exact(lind,model,model2,parameters,iters,start_learning_rate)[0]
        distt.append(overlap_exact(lind,model,model2,params,parameters))
        parameters = params
        t += dt

        tt.append(t)
        pt.append(parameters)

    return tt,pt,distt



def ptVMC_implicit_exact(lind,model,parameters,stop_time,dt,iters=1000,start_learning_rate=1e-3,order=1):
    # Euler loop by optimizing exacly at each time step, return the parameters and distance reached at each time step

    model2 = model.__class__() # only works if classes used have default values for all their arguments
    model2.__dict__ = copy(model.__dict__) # shallow copy because deepcopy doesn't work on hilbert spaces. Must be careful
    model2.channel = small_evol(lind, dt/2, order=order)

    model.channel = small_evol(lind, -dt/2, order=order)

    tt = [0.]
    pt = [parameters]
    distt = [jnp.array(-1)]

    t = 0.
    for _ in range(int(stop_time/dt)):
        params = optimize_exact(lind,model,model2,parameters,iters,start_learning_rate)[0]
        distt.append(overlap_exact(lind,model,model2,params,parameters))
        parameters = params
        t += dt

        tt.append(t)
        pt.append(parameters)

    model.channel = None # undo the changes to model, useful when working with a notebook

    return tt,pt,distt