"""
QR-DQN Agent

Quantile regression DQN agent.

Features:
- Double DQN
- Dueling DQN
- Global grad norm clipping
- Vectorised environments
- Soft target network updates
"""

from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.struct import dataclass, field
from flax.linen.initializers import orthogonal
import optax
from chex import Scalar, Array, ArrayTree, PRNGKey

from ..common.networks import MLP, SimpleCNN
from ..common.utils import AgentState
from .dqn import DQN, DQNState


### Network ###

class QuantileQNetwork(nn.Module):
    """Network for estimatating a quantile distribution of Q-values."""
    
    action_dim: int
    pixel_obs: bool
    hidden_dims: Sequence[int]
    num_quantiles: int
    dueling: bool = True

    @nn.compact
    def __call__(self, observations: Array) -> Array:
        # Use CNN for pixel observations
        if self.pixel_obs:
            torso = SimpleCNN()
        else:
            torso = lambda x: x
        x = torso(observations)

        # MLP core
        x = MLP(self.hidden_dims)(x)

        # Dueling network architecture
        if self.dueling:
            advantages = nn.Dense(self.action_dim * self.num_quantiles, kernel_init=orthogonal(1.0))(x)
            advantages = advantages.reshape(*x.shape[:-1], self.action_dim, self.num_quantiles)
            value = nn.Dense(self.num_quantiles, kernel_init=orthogonal(1.0))(x)
            value = value.reshape(*x.shape[:-1], 1, self.num_quantiles)
            q_values = value + (advantages - advantages.mean(axis=1, keepdims=True))
        else:
            x = nn.Dense(self.action_dim * self.num_quantiles, kernel_init=orthogonal(1.0))(x)
            q_values = x.reshape(*x.shape[:-1], self.action_dim, self.num_quantiles)
       
        return q_values


### Agent State ###

# Alias
QRDQNState = DQNState


### Agent ###

@dataclass
class QRDQN(DQN):
    """Quantile regression DQN agent."""

    num_quantiles: int = field(False, default=19)  # Number of predicted quantiles
    kappa: float       = field(True, default=1.0)  # Huber loss kappa

    def create_agent_state(
        self,
        key: PRNGKey
    ) -> AgentState:
        """Initialise network, parameters and optimiser."""

        # Create network
        action_dim = self.action_space.n
        obs_shape = self.observation_space.shape
        sample_obs = self.observation_space.sample(jax.random.PRNGKey(0))
        if len(obs_shape) not in (1, 3):
            raise Exception(f"Invalid observation space shape: {obs_shape}.")
        pixel_obs = len(obs_shape) == 3
        network = QuantileQNetwork(
            action_dim, pixel_obs, self.hidden_dims, self.num_quantiles, self.dueling
        )

        # Set learning rate
        learning_rate = optax.linear_schedule(
            init_value=self.learning_rate,
            end_value=0.0,
            transition_steps=self.num_rollouts,
        ) if self.anneal_lr else self.learning_rate
        
        # Configure optimiser with optional gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=learning_rate, eps=1e-8)
        )

        # Create and return AgentState
        return QRDQNState.create(
            apply_fn=network.apply,
            params=network.init(key, sample_obs[None, ...]),
            target_params=network.init(key, sample_obs[None, ...]),
            epsilon=self.epsilon_start,
            tx=optimizer
        )

    def select_action(
        self, 
        key: PRNGKey, 
        agent_state: AgentState,
        observations: Array
    ):
        """Action selection logic."""

        # RNG
        key_epsilon, key_action = jax.random.split(key)
        
        # Forward pass through Q-network
        q_values = agent_state.apply_fn(agent_state.params, observations).mean(axis=2)
        
        # Epsilon-greedy action selection
        num_envs, action_dim = q_values.shape
        actions = jnp.where(
            jax.random.uniform(key_epsilon, shape=num_envs) > agent_state.epsilon,
            q_values.argmax(axis=-1),
            jax.random.randint(
                key_action, shape=num_envs, minval=0, maxval=action_dim
            )
        )
        return {'actions': actions}
    
    def learn(
            self,
            agent_state: AgentState, 
            batch: ArrayTree
    ) -> AgentState:
        """Update agent parameters with a batch of experience."""

        def quantile_huber_loss(params: ArrayTree) -> Scalar:
            """Differentiable TD-error loss function."""

            # Q-values for current observations (B, A, N)
            state_q = agent_state.apply_fn(params, batch.observations)

            # Select Q-values for taken actions (B, N)
            action_q = state_q[jnp.arange(self.batch_size), batch.actions, :]

            # Compute TD-error and broadcast (B, N, N)
            td_error = target_q[:, None, :] - action_q[:, :, None]

            # Calculate Huber loss (B, N, N)
            huber_loss = jnp.where(
                jnp.abs(td_error) <= self.kappa, 
                0.5 * (td_error ** 2),
                self.kappa * (jnp.abs(td_error) - 0.5 * self.kappa)
            ) 

            # Determine quantiles (1, 1, N)
            taus = jnp.linspace(0.0, 1.0, num=self.num_quantiles + 2)[None, None, 1:-1]

            # Calculate quantile Huber loss (B, N, N)
            quantile_loss = jnp.abs(taus - (td_error < 0).astype(jnp.float32)) * huber_loss 

            # Aggregate loss (scalar)
            loss = quantile_loss.sum(axis=1).mean(axis=1).mean()

            return loss
        
        # Q-values for next observations using target network (B, A, N)
        next_state_q = agent_state.apply_fn(agent_state.target_params, batch.next_observations)

        # Double DQN selects next actions with online network (B,)
        next_state_actions = agent_state.apply_fn(
            agent_state.params, batch.next_observations
        ).mean(axis=2).argmax(axis=1)

        # Gather Q-values for the selected actions from target network (B, N)
        next_action_q = next_state_q[jnp.arange(self.batch_size), next_state_actions, :]
        
        # Compute target Q-values with Bellman equation (B, N)
        target_q = batch.rewards[:, None] + \
            self.gamma * (1.0 - batch.terminations[:, None]) * next_action_q

        # Compute quantile Huber loss and gradients
        loss, grads = jax.value_and_grad(quantile_huber_loss)(agent_state.params)

        # Update model parameters with gradients
        agent_state = agent_state.apply_gradients(grads=grads)

        # Return updated agent state
        return agent_state
    