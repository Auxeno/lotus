from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from flax.struct import dataclass


def _as_sample_shape(sample_shape: Sequence[int] | int | None) -> Tuple[int, ...]:
    """Normalise sample_shape inputs to a tuple for broadcasting."""
    if sample_shape is None:
        return ()
    if isinstance(sample_shape, int):
        return (sample_shape,)
    return tuple(sample_shape)


@dataclass
class Categorical:
    """Categorical distribution parametrised by logits."""

    logits: Array
    axis: int = -1

    @property
    def _log_probs(self) -> Array:
        return jax.nn.log_softmax(self.logits, axis=self.axis)

    @property
    def probs(self) -> Array:
        return jnp.exp(self._log_probs)

    def sample(
        self, seed: PRNGKey, sample_shape: Sequence[int] | int | None = None
    ) -> Array:
        """Draw samples from the categorical distribution."""
        sample_shape = _as_sample_shape(sample_shape)
        logits = self.logits
        if sample_shape:
            broadcast_shape = sample_shape + logits.shape
            logits = jnp.broadcast_to(logits, broadcast_shape)
        axis = self.axis
        if sample_shape and axis >= 0:
            axis += len(sample_shape)
        return jax.random.categorical(seed, logits, axis=axis)

    def log_prob(self, actions: Array) -> Array:
        """Return log probability of provided categorical indices."""
        actions = jnp.asarray(actions, dtype=jnp.int32)
        gather_axis = self.axis
        expanded_actions = jnp.expand_dims(actions, gather_axis)
        gathered = jnp.take_along_axis(
            self._log_probs, expanded_actions, axis=gather_axis
        )
        return jnp.squeeze(gathered, axis=gather_axis)

    def entropy(self) -> Array:
        """Shannon entropy of the categorical distribution."""
        log_probs = self._log_probs
        probs = jnp.exp(log_probs)
        return -jnp.sum(probs * log_probs, axis=self.axis)

    def mode(self) -> Array:
        """Most likely class index."""
        return jnp.argmax(self.logits, axis=self.axis)

    def mean(self) -> Array:
        """Return categorical probabilities, matching distrax semantics."""
        return self.probs


@dataclass
class Normal:
    """Diagonal Normal distribution parametrised by location and scale."""

    loc: Array
    scale: Array

    @property
    def _dtype(self):
        return jnp.result_type(self.loc, self.scale)

    def _broadcast_to_value(self, value: Array) -> tuple[Array, Array, Array]:
        loc, scale, value = jnp.broadcast_arrays(self.loc, self.scale, value)
        return loc, scale, value

    def sample(
        self, seed: PRNGKey, sample_shape: Sequence[int] | int | None = None
    ) -> Array:
        """Draw reparameterised samples from the Normal distribution."""
        sample_shape = _as_sample_shape(sample_shape)
        event_shape = jnp.broadcast_shapes(self.loc.shape, self.scale.shape)
        full_shape = sample_shape + event_shape
        dtype = self._dtype
        eps = jax.random.normal(seed, full_shape, dtype=dtype)
        loc = jnp.broadcast_to(self.loc, full_shape).astype(dtype)
        scale = jnp.broadcast_to(self.scale, full_shape).astype(dtype)
        return loc + scale * eps

    def log_prob(self, value: Array) -> Array:
        """Return element-wise log probability density."""
        loc, scale, value = self._broadcast_to_value(value)
        dtype = self._dtype
        loc = loc.astype(dtype)
        scale = scale.astype(dtype)
        value = value.astype(dtype)
        log_two_pi = jnp.log(jnp.array(2.0 * jnp.pi, dtype=dtype))
        squared_term = jnp.square((value - loc) / scale)
        return -0.5 * (log_two_pi + 2.0 * jnp.log(scale) + squared_term)

    def entropy(self) -> Array:
        """Differential entropy of the Normal distribution."""
        _, scale = jnp.broadcast_arrays(self.loc, self.scale)
        dtype = self._dtype
        scale = scale.astype(dtype)
        log_two_pi_e = 0.5 * jnp.log(jnp.array(2.0 * jnp.pi * jnp.e, dtype=dtype))
        return log_two_pi_e + jnp.log(scale)

    def mean(self) -> Array:
        return self.loc.astype(self._dtype)

    def stddev(self) -> Array:
        return self.scale.astype(self._dtype)

    def variance(self) -> Array:
        scale = self.scale.astype(self._dtype)
        return jnp.square(scale)

    def mode(self) -> Array:
        return self.loc.astype(self._dtype)
