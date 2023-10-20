from functools import partial

import jax
import numpy as np
from jax import numpy as jnp, jit
from jax import random as jr
from numpyro import distributions as dist
from numpyro.distributions import constraints


@partial(jit, static_argnums=(1,))
def moving_window(a, size: int):
    starts = jnp.arange(len(a) - size + 1)
    return jax.vmap(lambda start: jax.lax.dynamic_slice(a, (start,), (size,)))(starts)


# pylint: disable=too-many-instance-attributes,duplicate-code
class Autoregressive(dist.Distribution):
    """
    An autoregressive model.

    Attributes
    ----------
    parameters: jnp.Array
        an initializer object from Flax
    parameters: Optional[jnp.Array]
        an initializer object from Flax
    """

    arg_constraints = {
        "loc": constraints.real,
        "ar_coefficients": constraints.real_vector,
        "scale": constraints.positive,
    }
    support = constraints.real_vector
    reparametrized_params = ["loc", "scale", "ar_coefficients"]

    def __init__(self, loc, ar_coefficients, scale, length=None):
        super().__init__()
        self.loc = loc
        self.ar_coefficients = ar_coefficients
        self.scale = scale
        self.p = len(ar_coefficients)
        self.length = length

    def sample(self, rng_key, length=None, initial_state=None, sample_shape=()):
        if length is None:
            length = self.length

        def body_fn(states, sample_key):
            states = jnp.atleast_1d(states)
            take = np.minimum(self.p, states.shape[0])
            lhs = jax.lax.dynamic_slice_in_dim(
                states, states.shape[0] - take, take
            )[::-1]
            rhs = jax.lax.dynamic_slice_in_dim(self.ar_coefficients, 0, take)
            loc = self.loc + jnp.einsum("i,i->", lhs, rhs)
            yt = jnp.atleast_1d(dist.Normal(loc, self.scale).sample(sample_key))
            states = jnp.concatenate([states, yt], axis=-1)
            return states

        sample_keys = jr.split(rng_key, length)
        if initial_state is None:
            initial_state = dist.Normal(self.loc, self.scale).sample(
                sample_keys[0]
            )
        states = jnp.atleast_1d(initial_state)
        for sample_key in sample_keys[1:]:
            states = body_fn(states, sample_key)
        return states

    def log_prob(self, value):
        states = jnp.atleast_1d(value)
        rev_states_padded = jnp.concatenate([np.zeros(self.p), states])[::-1]
        seqs = moving_window(rev_states_padded, self.p + 1)
        seqs = seqs[:states.shape[0]]
        locs = self.loc + jnp.einsum(
            "ji,i->j", seqs[:, 1:], self.ar_coefficients
        )
        lp = dist.Normal(locs, self.scale).log_prob(seqs[:, 0])
        return jnp.sum(lp)

    def mean(self, length=None, initial_state=None):
        if length is None:
            length = self.length

        def body_fn(states):
            states = jnp.atleast_1d(states)
            take = np.minimum(self.p, states.shape[0])
            lhs = jax.lax.dynamic_slice_in_dim(
                states, states.shape[0] - take, take
            )[::-1]
            rhs = jax.lax.dynamic_slice_in_dim(self.ar_coefficients, 0, take)
            loc = jnp.atleast_1d(self.loc + jnp.einsum("i,i->", lhs, rhs))
            states = jnp.concatenate([states, loc], axis=-1)
            return states

        if initial_state is None:
            initial_state = jnp.atleast_1d(self.loc)
        states = jnp.atleast_1d(initial_state)
        for _ in range(1, length):
            states = body_fn(states)
        return states


