import warnings

import jax.numpy as jnp

from .algos import DDPG, DQN, PPO, PQN, QRDQN, RPPO, SAC, TD3

# Suppress warning caused by Gymnax
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*scatter inputs have incompatible types.*",
)


# Update dtype used by Gymnax spaces
jnp.int_ = jnp.int32

__all__ = [
    "DQN",
    "QRDQN",
    "PQN",
    "DDPG",
    "TD3",
    "SAC",
    "PPO",
    "RPPO",
]
