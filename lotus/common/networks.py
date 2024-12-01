"""
Networks Module

Provides network architectures for agents with Flax.
Features:
- MLP network
- Simple CNN torso network
"""

from typing import Sequence, Callable
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, he_normal
from chex import Array


class MLP(nn.Module):
    """Simple MLP network."""
    
    hidden_dims: Sequence[int]
    activation_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x: Array) -> Array:        
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(x)
            x = self.activation_fn(x)
        return x
    
    
class SimpleCNN(nn.Module):
    """Simple CNN torso network for pixel observations."""
    
    activation_fn: Callable = nn.relu
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            kernel_init=he_normal()
        )(x)
        x = self.activation_fn(x)
        x = x.reshape(*x.shape[:-3], -1)
        return x