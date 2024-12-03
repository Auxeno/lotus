import warnings
import jax.numpy as jnp

from .algos import DQN, QRDQN, DDPG, PPO


# Suppress warning caused by Gymnax
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*scatter inputs have incompatible types.*",
)


# Update dtype used by Gymnax spaces
jnp.int_ = jnp.int32