"""
Networks Module

Provides network architectures for agents with Flax.
Features:
- MLP network
- Simple CNN torso network
"""

from typing import Tuple, Sequence, Callable
from functools import partial
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, he_normal
from chex import Array


class MLP(nn.Module):
    """Simple MLP network."""
    
    hidden_dims: Sequence[int]
    activation_fn: Callable = nn.relu
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x: Array) -> Array:       
        if self.layer_norm:
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(x)
            x = normalize(x)
            x = self.activation_fn(x)
        return x
    
    
class SimpleCNN(nn.Module):
    """Simple CNN torso network for pixel observations."""
    
    activation_fn: Callable = nn.relu
    layer_norm: bool = False
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        if self.layer_norm:
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            kernel_init=he_normal()
        )(x)
        x = normalize(x)
        x = self.activation_fn(x)
        x = x.reshape(*x.shape[:-3], -1)
        return x
    

class GRUCore(nn.Module):
    """Scanned GRU core."""
    
    @partial(
        nn.scan,
        variable_broadcast='params',
        in_axes=0,
        out_axes=0,
        split_rngs={'params': False},
    )
    @nn.compact
    def __call__(self, rnn_state: Array, x: Tuple[Array, Array]):
        # Unpack carry
        inputs, resets = x
        
        # Reset hidden state for reset flags
        rnn_state = jnp.where(
            resets[:, None],
            self.initialize_carry(inputs.shape[0], inputs.shape[1]),
            rnn_state,
        )
        
        # Forward pass through GRU cell
        new_rnn_state, y = nn.GRUCell(features=inputs.shape[1])(rnn_state, inputs)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )
    