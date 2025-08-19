"""
Utilities Module

Features:
- Environment transition
- AgentState alias
- Logs
"""
from chex import Scalar, Array
from flax.struct import dataclass, field
from flax.training.train_state import TrainState


@dataclass
class Transition:
    """Transition for a single step in vectorised environments."""
    observations: Array = field(pytree_node=True)
    next_observations: Array = field(pytree_node=True)
    actions: Array = field(pytree_node=True)
    rewards: Array = field(pytree_node=True)
    terminations: Array = field(pytree_node=True)
    truncations: Array = field(pytree_node=True)


AgentState = TrainState  # alias


@dataclass
class Logs:
    rewards: Array = field(pytree_node=True)
    dones: Array = field(pytree_node=True)
    global_step: Scalar | None = field(pytree_node=True, default=None)
