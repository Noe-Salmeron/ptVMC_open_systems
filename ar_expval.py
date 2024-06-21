import netket as nk

import jax
import jax.numpy as jnp
from jax.nn.initializers import normal

from functools import partial

###

# Functions to compute the expectation values of observables using the combination of AR wavefunctions as an ansatz.
# This is done by sampling from each wavefunctions istead of sampling from the trace of the density matrix.

###


default_kernel_init = normal(stddev=0.001)
default_bias_init = normal(stddev=0.001)
def arc_vstate_list(arc_model, parameters, n_samples=100):
    # returns samples for each ar wavefunction of the ansatz. The shape of the output is (rank, n_batches, n_samples, n_spin)
    ar = nk.models.ARNNDense(
            hilbert = arc_model.hilbert,
            layers = arc_model.layers, 
            features = arc_model.features, 
            param_dtype = arc_model.param_dtype,
            kernel_init = default_kernel_init,
            bias_init = default_bias_init
        )
    
    sa = nk.sampler.ARDirectSampler(ar.hilbert) # type:ignore

    vstate_list = [nk.vqs.MCState(sa, ar, n_samples=n_samples, seed=1) for j in range(arc_model.rank)]
    for j in range(arc_model.rank):
        vstate_list[j].parameters = parameters["params"][f"ar_{j}"]

    return vstate_list


@partial(jax.jit, static_argnums=[0,2])
def arc_sample(arc_model, parameters, n_samples=100):
    vstate_list = arc_vstate_list(arc_model, parameters, n_samples)
    return jnp.array([jnp.array(vstate_list[j].samples) for j in range(arc_model.rank)])


@partial(jax.jit, static_argnums=[1,3]) # n_samples is not jittable bacause it is part of an if statement in the netket code
def arc_expval(local_observable, arc_model, parameters, n_samples=100):
    vstate_list = arc_vstate_list(arc_model, parameters, n_samples)
    expect_list = jnp.array([vstate_list[j].expect(local_observable).mean for j in range(arc_model.rank)])

    p = parameters["params"]["p"]
    p = jnp.abs(p)
    p = p / jnp.sum(p)

    return jnp.dot(p, expect_list)