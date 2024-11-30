"""
Replay Buffer

Efficiently stores and samples experiences for JAX agents.
Optimised for high performance with JIT compilation and donate_argnums.
"""

from functools import partial
import jax
import jax.numpy as jnp
from flax.struct import dataclass, field
from chex import Scalar, ArrayTree, PRNGKey


@dataclass
class BufferState:
    data: ArrayTree   = field(pytree_node=True)  # Pytree with leading dimensions (max_steps, num_envs)
    idx: Scalar       = field(pytree_node=True)  # Next insertion index
    size: Scalar      = field(pytree_node=True)  # Current number of items in buffer
    num_envs: Scalar  = field(pytree_node=True)  # Number of vectorised environments
    max_size: Scalar  = field(pytree_node=True)  # Maximum buffer capacity


class Buffer:
    @staticmethod
    def init_buffer(
        sample_item: ArrayTree, 
        num_envs: Scalar,
        max_size: Scalar
    ) -> BufferState:
        """Initialise replay buffer with a given capacity."""
        assert max_size % num_envs == 0, "max_size must be divisible by num_envs"
        def allocate(x):
            return jnp.zeros((max_size // num_envs, num_envs) + x.shape, dtype=x.dtype)

        data = jax.tree.map(allocate, sample_item)
        return BufferState(
            data=data,
            idx=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
            num_envs=jnp.array(num_envs, dtype=jnp.int32),
            max_size=jnp.array(max_size, dtype=jnp.int32)
        )

    @staticmethod
    @partial(jax.jit, donate_argnums=(0,))
    def push(
        buffer_state: BufferState, 
        item: ArrayTree
    ) -> BufferState:
        """Add a single item to replay buffer."""
        # Insert the item at the current index position
        new_data = jax.tree.map(
            lambda data, item_elem: data.at[buffer_state.idx].set(item_elem),
            buffer_state.data,
            item
        )
        # Update index and size
        new_idx = (buffer_state.idx + 1) % (buffer_state.max_size // buffer_state.num_envs)
        new_size = jnp.minimum(buffer_state.size + buffer_state.num_envs, buffer_state.max_size)

        return BufferState(
            data=new_data,
            idx=new_idx,
            size=new_size,
            num_envs=buffer_state.num_envs,
            max_size=buffer_state.max_size
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def sample(
        key: PRNGKey, 
        buffer_state: BufferState, 
        batch_size: Scalar
    ) -> ArrayTree:
        """Uniformly sample a batch of items from the replay buffer across all environments."""
        key_step, key_env = jax.random.split(key)

        # Sample step indices uniformly from [0, num_steps)
        step_indices = jax.random.randint(key_step, shape=(batch_size,), 
                                          minval=0, maxval=buffer_state.size // buffer_state.num_envs)

        # Sample environment indices uniformly from [0, num_envs)
        env_indices = jax.random.randint(key_env, shape=(batch_size,), 
                                         minval=0, maxval=buffer_state.num_envs)

        # Sample the data using sampled step and environment indices
        sampled_data = jax.tree.map(
            lambda data: data[step_indices, env_indices],
            buffer_state.data
        )

        return sampled_data
    