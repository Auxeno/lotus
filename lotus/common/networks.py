"""
Networks Module

Provides network architectures for agents with Flax.
Features:
- MLP and CNN torso networks
"""

from typing import Sequence
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, he_normal
from chex import Array


class MLPTorso(nn.Module):
    """MLP torso network for vector observations."""
    
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x: Array) -> Array:        
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(x)
            x = nn.relu(x)
        return x
    
    
class SimpleCNNTorso(nn.Module):
    """Simple CNN torso network for pixel observations."""
    
    hidden_dims: Sequence[int]
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            kernel_init=he_normal()
        )(x)
        x = nn.relu(x)
        x = x.reshape(*x.shape[:-3], -1)
        x = MLPTorso(self.hidden_dims, self.layer_norm)(x)
        return x