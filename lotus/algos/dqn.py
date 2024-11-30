"""
DQN Agent

Basic DQN agent.

Features:
- Double DQN
- Dueling DQN
- Global grad norm clipping
- Vectorised environments
- Soft target network updates
"""

from typing import Any, Tuple, Dict, Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.struct import dataclass, field
from flax.linen.initializers import orthogonal
import optax
from chex import Scalar, Array, ArrayTree, PRNGKey

from ..common.agent import BaseAgent
from ..common.networks import MLPTorso, SimpleCNNTorso
from ..common.buffer import Buffer
from ..common.utils import Transition, AgentState, Logs


### Network ###

class QNetwork(nn.Module):
    """Q-Network with configurable torso and dueling architecture."""
    
    action_dim: int
    pixel_obs: bool
    hidden_dims: Sequence[int]
    dueling: bool = True

    @nn.compact
    def __call__(self, observations: Array) -> Array:
        # Select and initialise torso
        if self.pixel_obs:
            torso = SimpleCNNTorso(self.hidden_dims)
        else:
            torso = MLPTorso(self.hidden_dims)
        x = torso(observations)

        # Dueling network architecture
        if self.dueling:
            advantages = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0))(x)
            value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
            q_values = value + (advantages - advantages.mean(axis=-1, keepdims=True))
        else:
            q_values = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0))(x)
        
        return q_values
    

### Agent State ###

class DQNState(AgentState):
    target_params: ArrayTree = field(True)


### Agent ###

@dataclass
class DQN(BaseAgent):

    batch_size: int         = field(False, default=64)
    dueling: bool           = field(False, default=True)
    learning_starts: int    = field(False, default=1000)
    buffer_capacity: int    = field(False, default=100_000)
    target_update_freq: int = field(False, default=1)
    tau: float              = field(True, default=0.05)
    epsilon_start: float    = field(True, default=0.5)
    epsilon_final: float    = field(True, default=0.05)
    epsilon_fraction: float = field(True, default=0.8)

    def create_agent_state(
        self,
        key: PRNGKey
    ) -> AgentState:
        """Initialise AgentState for algorithm."""

        # Create network
        action_dim = self.action_space.n
        obs_shape = self.observation_space.shape
        if len(obs_shape) not in (1, 3):
            raise Exception(f"Invalid observation space shape: {obs_shape}.")
        pixel_obs = len(obs_shape) == 3
        network = QNetwork(
            action_dim, pixel_obs, self.hidden_dims, self.dueling
        )

        # Initialise network parameters
        sample_obs = self.observation_space.sample(jax.random.PRNGKey(0))
        params = network.init(key, sample_obs[None, ...])
        target_params = network.init(key, sample_obs[None, ...])
        
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

        # Create and return DQNAgentState
        return DQNState.create(
            apply_fn=network.apply,
            params=params,
            target_params=target_params,
            tx=optimizer
        )

    def init_train_carry(
        self,
        rng: PRNGKey
    ) -> Dict:

        # RNG
        rng, key_agent, key_reset, key_rollout = jax.random.split(rng, 4)
        dummy_key = jax.random.PRNGKey(0)
        
        # Initialise agent state
        agent_state = self.create_agent_state(key_agent)

        # Initialise buffer with a sample transition
        sample_transition = Transition(
            observations=self.observation_space.sample(dummy_key),
            next_observations=self.observation_space.sample(dummy_key),
            actions=self.action_space.sample(dummy_key),
            rewards=jnp.array(1.0, dtype=jnp.float32),
            terminations=jnp.array(False, dtype=bool),
            truncations=jnp.array(False, dtype=bool)
        )
        buffer_state = Buffer.init_buffer(
            sample_transition, self.num_envs, self.buffer_capacity
        )

        # Initial observations and environment states
        reset_result = self.env_reset(key_reset)

        # Build initial rollout carry
        rollout_carry = {
            'key': key_rollout,
            'env_states': reset_result['env_states'],
            'observations': reset_result['observations']
        }

        # Initial logs
        logs = Logs(
            rewards=jnp.zeros((self.num_rollouts, self.rollout_steps,
                               self.num_envs), dtype=jnp.float32),
            dones=jnp.zeros((self.num_rollouts, self.rollout_steps, 
                             self.num_envs), dtype=bool),
            global_step=0
        )

        return {
            'rng': rng,
            'agent_state': agent_state,
            'buffer_state': buffer_state,
            'rollout_carry': rollout_carry,
            'global_step': 0,
            'logs': logs
        }

    def select_action(
        self, 
        key: PRNGKey, 
        agent_state: AgentState,
        observations: Array, 
        epsilon: Scalar
    ):
        """Action selection logic for algorithm."""

        # RNG
        key_epsilon, key_action = jax.random.split(key)
        
        # Forward pass through Q-network
        q_values = agent_state.apply_fn(agent_state.params, observations)
        
        # Epsilon-greedy action selection
        num_envs, action_dim = q_values.shape
        actions = jnp.where(
            jax.random.uniform(key_epsilon, shape=num_envs) > epsilon,
            q_values.argmax(axis=-1),
            jax.random.randint(
                key_action, shape=num_envs, minval=0, maxval=action_dim
            )
        )
        return {'actions': actions}

    def rollout(
        self,
        initial_carry: Dict,
        agent_state: AgentState,
        epsilon: Scalar
    ):
        """Agent performs a rollout in environment generating experience."""
        
        def rollout_step(carry: Dict, _: Any) -> Tuple[Dict, Transition]:
            """Scannable single vectorised environment step."""

            # Unpack carry
            key, env_states, observations = (
                carry['key'], carry['env_states'], carry['observations']
            )

            # RNG
            key, key_action, key_step = jax.random.split(key, 3)

            # Action selection
            actions = self.select_action(
                key_action, agent_state, observations, epsilon
            )['actions']

            # Environment step
            step_result = self.env_step(key_step, env_states, actions)

            # Build carry for next step
            new_carry = {
                'key': key,
                'env_states': step_result['next_env_states'],
                'observations': step_result['next_observations']
            }

            # Build transition
            transition = Transition(
                observations=observations,
                next_observations=step_result['next_observations'],
                actions=actions,
                rewards=step_result['rewards'],
                terminations=step_result['terminations'],
                truncations=step_result['truncations']
            )

            # Build logs for step
            dones = jnp.logical_or(step_result['terminations'], step_result['truncations'])
            logs = Logs(rewards=step_result['rewards'], dones=dones)

            return new_carry, (transition, logs)
            
        # Scan to generate a batch of transitions
        final_carry, (experiences, logs) = jax.lax.scan(
            f=rollout_step, init=initial_carry, xs=None, length=self.rollout_steps
        )

        # Return experiences, logs and final carry
        return {
            'experiences': experiences,
            'logs': logs,
            'carry': final_carry
        }

    def learn(
        self,
        agent_state: AgentState,
        batch: ArrayTree
    ) -> AgentState:
        """Agent learning step logic."""

        def td_error_loss(params: ArrayTree) -> Scalar:
            """Differentiable TD-error loss function."""
            
            # Predict Q-values for current observations
            state_q = agent_state.apply_fn(params, batch.observations)

            # Select Q-values for taken actions
            action_q = state_q[jnp.arange(self.batch_size), batch.actions]

            # Compute TD-error loss as mean squared error
            return ((action_q - target_q) ** 2).mean()

        # Predict Q-values for next observations using target network
        next_state_q = agent_state.apply_fn(agent_state.target_params, batch.next_observations)

        # Select maximum Q-value for next states
        next_action_q = next_state_q.max(axis=1)

        # Compute target Q-values using Bellman equation
        target_q = batch.rewards + self.gamma * (1.0 - batch.terminations) * next_action_q 

        # Compute TD-error loss and gradients
        loss, grads = jax.value_and_grad(td_error_loss)(agent_state.params)

        # Update model parameters with gradients
        agent_state = agent_state.apply_gradients(grads=grads)

        # Return updated agent state
        return agent_state

    def soft_update(
        self,
        online_params: ArrayTree, 
        target_params: ArrayTree, 
    ) -> ArrayTree:
        """Partially update target network parameters."""
        
        return jax.tree.map(
            lambda t, o: self.tau * o + (1.0 - self.tau) * t, target_params, online_params
        )

    def epsilon_decay(
        self,
        global_step: int
    ) -> Scalar:
        """Linear epsilon decay."""
        
        decay_steps = self.epsilon_fraction * self.total_steps
        epsilon = self.epsilon_start + (self.epsilon_final - self.epsilon_start) * \
            (global_step / decay_steps)
        return jnp.maximum(epsilon, self.epsilon_final)

    def train(
        self,
        seed: int,
    ) -> Dict:
        
        def train_step(carry: Dict, _: Any) -> Tuple[Dict, None]:
            """Scannable single train step."""

            # Unpack carry
            rng, agent_state, buffer_state, rollout_carry, global_step, logs = (
                carry['rng'], 
                carry['agent_state'], 
                carry['buffer_state'], 
                carry['rollout_carry'], 
                carry['global_step'],
                carry['logs']
            )

            # RNG
            rng, key_rollout, key_sample = jax.random.split(rng, 3)

            # Calculate current epsilon
            epsilon = jax.lax.cond(
                buffer_state.size >= max(self.batch_size, self.learning_starts),
                lambda: self.epsilon_decay(global_step),
                lambda: jnp.array(1.0, dtype=jnp.float32)
            )

            # Generate experience batch
            rollout_result = self.rollout(rollout_carry, agent_state, epsilon)

            # Store experiences in buffer
            buffer_state, _ = jax.lax.scan(
                lambda buffer_state, experience: (Buffer.push(buffer_state, experience), None), 
                init=buffer_state, 
                xs=rollout_result['experiences']
            )

            # Perform learn step
            agent_state = jax.lax.cond(
                buffer_state.size >= max(self.batch_size, self.learning_starts),
                lambda: self.learn(
                    agent_state,
                    batch=Buffer.sample(key_sample, buffer_state, self.batch_size)
                ),
                lambda: agent_state
            )

            # Soft target network update
            agent_state = agent_state.replace(
                target_params=self.soft_update(agent_state.params, agent_state.target_params)
            )

            # Update logs
            steps_per_rollout = self.rollout_steps * self.num_envs
            global_step = global_step + steps_per_rollout
            logs = Logs(
                rewards=logs.rewards.at[global_step // steps_per_rollout].set(
                    rollout_result['logs'].rewards
                ),
                dones=logs.dones.at[global_step // steps_per_rollout].set(
                    rollout_result['logs'].dones
                ),
                global_step=global_step
            )

            # Print logs if verbose
            checkpoint = jnp.argmax(self.checkpoints == global_step) + 1
            jax.lax.cond(
                self.verbose and jnp.any(self.checkpoints == global_step),
                lambda: self.print_logs(logs, checkpoint),
                lambda: None,
            )

            # Build carry for next step
            new_carry = {
                'rng': rng,
                'agent_state': agent_state,
                'buffer_state': buffer_state,
                'rollout_carry': rollout_result['carry'],
                'global_step': global_step,
                'logs': logs
            }

            return new_carry, None

        # Initialise RNG
        rng = jax.random.PRNGKey(seed)

        # Initialise train carry
        initial_carry = self.init_train_carry(rng)

        # Scan the train step to train agent
        final_carry, _ = jax.lax.scan(
            f=train_step, init=initial_carry, xs=None, length=self.num_rollouts
        )
        return final_carry
    