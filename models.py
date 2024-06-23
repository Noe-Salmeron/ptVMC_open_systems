import netket as nk
from netket import nn as nknn
from netket.models.autoreg import ARNNSequential

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.nn.initializers import normal
from netket.utils.types import NNInitFunc
from math import factorial

from functools import partial
import flax.linen as nn

from typing import Union, Any

###

# Contains some ansatzes, multiple utility methods for manipulating these ansatzes, and some implementations of quantum channels.

###


def custom_initializer(key,shape,dtype=jnp.complex128):
    # initializes the complex numbers so that the phase is distributed uniformily when the exponential is taken
    stddev = 0.01
    if dtype == jnp.complex128:
        return jax.random.normal(key,shape,jnp.float64)*stddev + 1j * jax.random.uniform(key,shape,jnp.float64,-jnp.pi,jnp.pi)
    else:
        return jax.random.normal(key,shape,jnp.float64)*stddev
    

    
@partial(jax.jit, static_argnames=["model","lind"])
def to_dm(model,parameters,lind):
    # returns the non normalized density matrix
    all_configurations = jnp.array(lind.hilbert.all_states())
    dm = jnp.exp(model.apply(parameters,all_configurations))
    n = 2**(len(all_configurations[0])//2)
    return jnp.reshape(dm,(n,n))



def modulus_soft_max(arr, temperature=1):
    # softmax with the modulus of the complex numbers
    ret = jnp.exp(jnp.abs(arr) / temperature)
    return ret / jnp.sum(ret)



def n_params(lind,model):
    # return the number of parameters of the model
    key = jax.random.PRNGKey(1)
    confs = lind.hilbert.random_state(jax.random.PRNGKey(1),2)
    parameters = model.init(key, confs)
    return len(ravel_pytree(parameters)[0])



@jax.jit
def conf_to_index(conf):
    # takes a configuration and return its index, works for simple and doubled hilbert spaces
    l = jnp.size(conf, axis=-1)
    bits = (conf + 1) / 2
    powers = 2**jnp.arange(l-1,-1,-1)
    return jnp.int64(jnp.dot(bits, powers))



def tensor_identity(pauli_strings_operator, dim, side):
    # takes A and returns a PauliStrings of A\otimes I or I\otimes A
    if side == "left":
        ops = np.array([o.replace(o, "I"*dim + o) for o in pauli_strings_operator.operators])
    elif side == "right":
        ops = np.array([o.replace(o, o + "I"*dim) for o in pauli_strings_operator.operators])
    else:
        raise ValueError("side should be right or left")
    
    return nk.operator.PauliStrings(nk.hilbert.DoubledHilbert(nk.hilbert.Spin(s=1/2, N=dim)), ops, pauli_strings_operator.weights) # type: ignore



def lind_to_pauli_strings(lind):
    # returns a PauliStrings version of the liouvilian that can then be jitted
    ha = lind.hamiltonian
    L = ha.hilbert.size

    HI = tensor_identity(ha.to_pauli_strings(), L, side="right")
    IHt = tensor_identity(ha.transpose().collect().to_pauli_strings(), L, side="left")

    ops = -1j*(HI - IHt)

    for j in range(len(lind.jump_operators)):
        jump = lind.jump_operators[j]
        
        JI = tensor_identity(jump.to_pauli_strings(), L, side="right")
        IJs = tensor_identity(jump.conjugate().collect().to_pauli_strings(), L, side="left")
        JdJI = tensor_identity((jump.conjugate().transpose() @ jump).collect().to_pauli_strings(), L, side="right")
        IJtJs = tensor_identity((jump.transpose() @ jump.conjugate()).collect().to_pauli_strings(), L, side="left")

        ops += JI @ IJs - 0.5*(JdJI + IJtJs)

    return ops



def lind_to_pauli_strings_TFI_nd(lind, g, h):
    # returns the non-diagonal part of the liouvillian under a PauliStrings form
    hi = lind.hamiltonian.hilbert
    L = hi.size

    sx_part = nk.operator.LocalOperator(hi,dtype=complex)
    for j in range(L): # type: ignore
        sx_part += -h * nk.operator.spin.sigmax(hi,j)

    HI = tensor_identity(sx_part.to_pauli_strings(), L, side="right")
    IHt = tensor_identity(sx_part.transpose().collect().to_pauli_strings(), L, side="left") # type: ignore

    ops = -1j*(HI - IHt)

    for j in range(len(lind.jump_operators)):
        jump = lind.jump_operators[j]
        
        JI = tensor_identity(jump.to_pauli_strings(), L, side="right")
        IJs = tensor_identity(jump.conjugate().collect().to_pauli_strings(), L, side="left")

        ops += JI @ IJs

    return ops



def lind_to_pauli_strings_TFI_d(lind, g, J):
    # returns the diagonal part of the liouvillian under a PauliStrings form
    hi = lind.hamiltonian.hilbert
    L = hi.size

    sz_part = nk.operator.LocalOperator(hi,dtype=complex)
    for (i,j) in g.edges(): # type: ignore
        sz_part += -J * nk.operator.spin.sigmaz(hi,i) @ nk.operator.spin.sigmaz(hi,j)

    HI = tensor_identity(sz_part.to_pauli_strings(), L, side="right")
    IHt = tensor_identity(sz_part.transpose().collect().to_pauli_strings(), L, side="left") # type: ignore

    ops = -1j*(HI - IHt)

    for j in range(len(lind.jump_operators)):
        jump = lind.jump_operators[j]
        
        JdJI = tensor_identity((jump.conjugate().transpose() @ jump).collect().to_pauli_strings(), L, side="right")
        IJtJs = tensor_identity((jump.transpose() @ jump.conjugate()).collect().to_pauli_strings(), L, side="left")

        ops += -0.5*(JdJI + IJtJs)

    return ops



default_kernel_init = normal(stddev=0.001)
default_bias_init = normal(stddev=0.001)
p_initializer: Any = normal(stddev=0.01)
class AR_combination(nn.Module):
    # density matrix represented as a sum of weighted auto regresive wavefunctions
    rank: int = 1
    hilbert: Any = nk.hilbert.Spin(s=1/2, N=2) # dummy initialization
    layers: int = 2
    features: int = 10
    param_dtype: Any = jnp.complex128

    channel: Any = None

    def setup(self):
        self.ar_list = [nk.models.ARNNDense(
            name = f"ar_{j}",
            hilbert = self.hilbert,
            layers = self.layers, 
            features = self.features, 
            param_dtype = self.param_dtype,
            kernel_init = default_kernel_init,
            bias_init = default_bias_init
        )
        for j in range(self.rank)]

    @nn.compact
    def __call__(self, input):
        if self.channel is not None:
            xp, mels = self.channel.get_conn_padded_jax(input)
        else:
            xp = input

        sigma,eta = jnp.split(xp,2,axis=-1)

        shape_temp = sigma.shape # mumbo jumbo reshaping because the ar wavefunctions only accept inputs of size (batch,nspins)
        sigma = sigma.reshape((-1,shape_temp[-1]))
        eta = eta.reshape((-1,shape_temp[-1]))

        p = self.param("p",p_initializer,(self.rank,),jnp.complex128)
        p = jnp.abs(p)
        p = p / jnp.sum(p)

        # the log needs to be returned, this is a netket convention and becomes important when using the samplers
        x = jnp.log(sum([p[j] * jnp.exp(self.ar_list[j](sigma)) * jnp.conj(jnp.exp(self.ar_list[j](eta))) for j in range(self.rank)]))
        x = x.reshape(shape_temp[:-1]) # undoing the mumbo jumbo reshaping

        if self.channel is not None:
            return jax.scipy.special.logsumexp(x, axis=-1, b=mels)
        else:
            return x
        


# default_kernel_init = custom_initializer
default_mixed_bias_init = normal(stddev=0.001)
# default_mixed_bias_init = custom_initializer
class Nagy_ansatz(nn.Module):
    # density matrix ansatz in https://arxiv.org/abs/1902.09483v2

    param_dtype: Any = jnp.complex128
    """The dtype of the weights."""
    activation: Any = nknn.log_cosh
    """The nonlinear activation function."""
    alpha: Union[float, int] = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    beta: Union[float, int] = 1
    """feature density. Number of features equal to beta * input.shape[-1]"""
    use_hidden_bias: bool = True
    """if True uses a bias in the 2 outer dense layers."""
    use_hidden_bias_mixed: bool = True
    """if True uses a bias in the dense layer realizing the mixing."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    hidden_bias_mixed_init: NNInitFunc = default_mixed_bias_init
    """Initializer for the hidden bias realizing the mixing (must be real values)."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""

    channel: Any = None

    @nn.compact
    def __call__(self, input):
        if self.channel is not None:
            xp, mels = self.channel.get_conn_padded_jax(input)
        else:
            xp = input

        sigma,eta = jnp.split(xp,2,axis=-1)

        X = nn.Dense(
            name="X",
            features=int(self.alpha * sigma.shape[-1]),
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )

        sigma_part = self.activation(X(sigma))
        sigma_part = jnp.sum(sigma_part, axis=-1)

        eta_part = self.activation(jnp.conj(X(eta)))
        eta_part = jnp.sum(eta_part, axis=-1)

        W = nn.Dense(
            name="W",
            features=int(self.beta * sigma.shape[-1]),
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=False,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )

        mixed_sigma_part = W(sigma)
        mixed_eta_part = jnp.conj(W(eta))

        if self.use_hidden_bias_mixed:
            h_bias = self.param(
                "c",
                self.hidden_bias_mixed_init,
                (int(self.beta * sigma.shape[-1]),),
                self.param_dtype, # should be float but we take the norm after
            )
            mixed_part = self.activation(mixed_eta_part + mixed_sigma_part + jnp.abs(h_bias)**2)
        else:
            mixed_part = self.activation(mixed_eta_part + mixed_sigma_part)
        mixed_part = jnp.sum(mixed_part, axis=-1)

        x = sigma_part + eta_part + mixed_part

        if self.use_visible_bias:
            v_bias = self.param(
                "a",
                self.visible_bias_init,
                (sigma.shape[-1],),
                self.param_dtype,
            )
            out_bias = jnp.dot(sigma, v_bias) + jnp.conj(jnp.dot(eta, v_bias))
            x = x + out_bias

        if self.channel is not None:
            return jax.scipy.special.logsumexp(x, axis=-1, b=mels)
        else:
            return x


# default_param_init = custom_initializer
default_param_init = default_kernel_init
class CC_RBM(nn.Module):
    # convex combination of RBM wavefunctions
    rank: int = 1
    alpha: float = 1.0
    param_dtype: Any = jnp.complex128

    channel: Any = None

    def setup(self):
        self.rbms = tuple(nk.models.RBM(alpha = self.alpha,
                                        param_dtype=self.param_dtype,
                                        kernel_init=default_param_init,
                                        hidden_bias_init=default_param_init,
                                        visible_bias_init = default_param_init) for i in range(self.rank))
        
    @nn.compact
    def __call__(self, input):
        if self.channel is not None:
            xp, mels = self.channel.get_conn_padded_jittable_but_terrible(input)
        else:
            xp = input

        sigma,eta = jnp.split(xp,2,axis=-1)
        # the log needs to be returned, this is a netket convention and becomes important when using the samplers
        x = jnp.log(sum([jnp.exp(self.rbms[i](sigma))*jnp.conj(jnp.exp(self.rbms[i](eta))) for i in range(self.rank)]))

        if self.channel is not None:
            return jax.scipy.special.logsumexp(x, axis=-1, b=mels)
        else:
            return x



class RBM_dm(nn.Module):
    # an RBM representing a density matrix

    param_dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.log_cosh
    """The nonlinear activation function."""
    alpha: Union[float, int] = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""

    channel: Any = None


    @nn.compact
    def __call__(self, input):

        if self.channel is not None:
            xp, mels = self.channel.get_conn_padded_jax(input)
        else:
            xp = input

        x = nn.Dense(
            name="Dense",
            features=int(self.alpha * input.shape[-1]),
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )(xp)
        x = self.activation(x)
        x = jnp.sum(x, axis=-1)

        if self.use_visible_bias:
            v_bias = self.param(
                "visible_bias",
                self.visible_bias_init,
                (input.shape[-1],),
                self.param_dtype,
            )
            out_bias = jnp.dot(xp, v_bias)
            x = x + out_bias
        
        if self.channel is not None:
            return jax.scipy.special.logsumexp(x, axis=-1, b=mels)
        else:
            return x
        


class Full_dm(nn.Module):
    # the full vectorized density matrix

    param_dtype: Any = np.complex128
    """The dtype of the weights."""
    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""

    channel: Any = None


    @nn.compact
    def __call__(self, input):

        if self.channel is not None:
            xp, mels = self.channel.get_conn_padded_jax(input)
        else:
            xp = input

        elements = self.param(
                "elements",
                self.kernel_init,
                (2 ** input.shape[-1],),
                self.param_dtype,
            )

        idx = conf_to_index(xp)
        x = elements[idx]
        
        if self.channel is not None:
            return jax.scipy.special.logsumexp(x, axis=-1, b=mels)
        else:
            return x


class small_evol(): # could be a subclass of nk.operator._abstract_super_operator.AbstractSuperOperator
    # class encoding the Taylor expansion, corresponding to a small evolution in time

    def __init__(self, lind, dt=0.01, order=1, id_fact=1.+0j, l_fact=1.+0j):
        self.lind = lind
        self.dt = dt
        self.order = order
        self.id_fact = id_fact
        self.l_fact = l_fact
        if type(lind) == nk.operator.PauliStringsJax:
            exp_terms = [(dt**j)/factorial(j) * self.operator_power(lind, j) for j in range(1,order+1)]
            self.ch = (id_fact * nk.operator.spin.identity(lind.hilbert).to_pauli_strings().to_jax_operator() +  # type: ignore
                       l_fact * sum(exp_terms)) # type: ignore
            self.ch._setup()
    
    def operator_power(self, op, power):
        if power == 1:
            return op
        else:
            return op @ self.operator_power(op, power-1)

    def get_conn_padded_numba(self,x): # x needs to be an array of array, only works for batches of configurations
        # in case self.lind is a numba operator, not jittable
        xp, mels = self.lind.get_conn_padded(x)
        xp = jnp.concatenate((xp, jnp.expand_dims(x, axis=1)), axis=-2)
        mels = jnp.concatenate((self.dt*mels, jnp.expand_dims(jnp.ones(len(x)), axis=-1)), axis=-1)
        return (xp,mels)
    
    def get_conn_padded_jax(self,x):
        return self.ch.get_conn_padded(x)
    
    def get_conn_padded_jittable_but_terrible(self,x):
        # returns all elements, not only the connected ones so it is not scalable but easy to implement and jittable.
        confs = self.lind.hilbert.all_states()
        l = jnp.array(self.lind.to_dense())
        indices = conf_to_index(x)
        xp = jnp.tile(confs,(len(x),1,1))
        mels = l[indices]

        xp = jnp.concatenate((xp, jnp.expand_dims(x, axis=1)), axis=-2)
        mels = jnp.concatenate((self.dt*mels, jnp.expand_dims(jnp.ones(len(x)), axis=-1)), axis=-1)
        return (xp,mels)
    
    def to_dense(self):
        l = self.lind.to_dense()
        return jnp.identity(jnp.size(l[0])) + self.dt*jnp.array(l)
    
    def to_dense_dag(self):
        return jnp.conj(jnp.transpose(self.to_dense()))
    


class exact_evol(): # could be a subclass of nk.operator._abstract_super_operator.AbstractSuperOperator, but deepcopy doesn't work
    # class encoding the operator exp(dt*L) corresponding to the exact evolution in time

    def __init__(self,lind,dt=0.01):
        self.lind = lind
        self.dt = dt
        self.ch = jax.scipy.linalg.expm(dt * lind.to_dense())
    
    def get_conn_padded_jax(self,x):
        confs = self.lind.hilbert.all_states()
        indices = conf_to_index(x)
        xp = jnp.tile(confs,(len(x),1,1))
        mels = self.ch[indices]

        return (xp,mels)
    
    def get_conn_padded_jittable_but_terrible(self,x):
        # returns all elements, not only the connected ones so it is not scalable but easy to implement and jittable.
        confs = self.lind.hilbert.all_states()
        l = jnp.array(self.lind.to_dense())
        indices = conf_to_index(x)
        xp = jnp.tile(confs,(len(x),1,1))
        mels = l[indices]

        xp = jnp.concatenate((xp, jnp.expand_dims(x, axis=1)), axis=-2)
        mels = jnp.concatenate((self.dt*mels, jnp.expand_dims(jnp.ones(len(x)), axis=-1)), axis=-1)
        return (xp,mels)
    
    def to_dense(self):
        return self.ch
    
    def to_dense_dag(self):
        return jnp.conj(jnp.transpose(self.to_dense()))