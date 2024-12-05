"""
PQN Agent

Parallised Q-Networks agent.

Features:
- Double DQN
- Dueling DQN
- Global grad norm clipping
"""

from typing import Any, Tuple, Dict, Sequence, Union
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.struct import dataclass, field
from flax.linen.initializers import orthogonal
import optax
from chex import Scalar, Array, ArrayTree, PRNGKey

from ..common.agent import OnPolicyAgent
from ..common.networks import MLP, SimpleCNN
from ..common.utils import AgentState, Transition, Logs


### Network ###

class QNetwork(nn.Module):
    """Network for estimatating Q-values."""
    
    action_dim: int
    pixel_obs: bool
    hidden_dims: Sequence[int]
    layer_norm: bool
    dueling: bool = True

    @nn.compact
    def __call__(self, observations: Array) -> Array:
        # Use CNN for pixel observations
        if self.pixel_obs:
            torso = SimpleCNN(layer_norm=self.layer_norm)
        else:
            torso = lambda x: x
        x = torso(observations)

        # MLP core
        x = MLP(self.hidden_dims, layer_norm=self.layer_norm)(x)

        # Dueling network architecture
        if self.dueling:
            advantages = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0))(x)
            value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
            q_values = value + (advantages - advantages.mean(axis=-1, keepdims=True))
        else:
            q_values = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0))(x)
        
        return q_values
    

### Agent State ###

class PQNState(AgentState):
    """State of a DQN agent, includes epsilon."""

    epsilon: Scalar = field(True)


### Environment Transition ###

@dataclass
class PQNTransition(Transition):
    """Extended transition for cleaner TD(λ) computation."""

    max_next_q: Union[Any, Array] = field(True, default=jnp.nan)


### Agent ###

@dataclass
class PQN(OnPolicyAgent):
    """Parallised Q-Network agent."""

    td_lambda: float        = field(True, default=0.5)       # Lambda value for TD(λ)
    layer_norm: bool        = field(False, default=True)     # Use layer norm in Q-network
    dueling: bool           = field(False, default=True)     # Dueling networks architecture
    epsilon_start: float    = field(True, default=0.5)       # Initial epsilon
    epsilon_final: float    = field(True, default=0.05)      # Final epsilon
    epsilon_fraction: float = field(True, default=0.8)       # Fraction of steps to decay

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
        network = QNetwork(
            action_dim, pixel_obs, self.hidden_dims, self.layer_norm, self.dueling
        )

        # Set learning rate
        learning_rate = optax.linear_schedule(
            init_value=self.learning_rate,
            end_value=0.0,
            transition_steps=self.num_rollouts,
        ) if self.anneal_lr else self.learning_rate
        
        # Configure optimiser with gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=learning_rate, eps=1e-8)
        )

        # Create and return AgentState
        return PQNState.create(
            apply_fn=network.apply,
            params=network.init(key, sample_obs[None, ...]),
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
        q_values = agent_state.apply_fn(agent_state.params, observations)
        
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
    
    def rollout(
        self,
        initial_carry: Dict,
        agent_state: AgentState,
    ):
        """Collect experience from environment."""
        
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
                key_action, agent_state, observations
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
            transition = PQNTransition(
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
            'carry': final_carry,
            'logs': logs
        }
    
    def calculate_lambda_returns(
        self,
        agent_state: PQNState,
        transitions: ArrayTree
    ):
        """Calculate returns using TD(λ)."""

        def lambda_step(lambda_return, transition):
            """Scannable TD(λ) step."""
            
            # Masks for non-terminal and non-truncated transitions
            non_termination = 1.0 - transition.terminations
            non_truncation = 1.0 - transition.truncations

            # Calculate bootstrapped return
            return_bootstrap = transition.max_next_q + \
                self.td_lambda * (lambda_return - transition.max_next_q)

            # Add reward and discount
            lambda_return = transition.rewards + \
                non_termination * self.gamma * return_bootstrap

            return non_truncation * lambda_return, lambda_return

        # Compute maximum next q for all next states
        max_next_q = agent_state.apply_fn(
            agent_state.params, transitions.next_observations
        ).max(axis=2)

        # Include max next q in transitions
        transitions = transitions.replace(max_next_q=max_next_q)

        # Initial lambda return
        initial_lambda_return = jnp.zeros(self.num_envs, dtype=jnp.float32)

        # Compute lambda returns with inversed scan
        _, lambda_returns = jax.lax.scan(
            lambda_step,
            initial_lambda_return,
            transitions,
            reverse=True
        )

        return lambda_returns

    def learn(
        self,
        agent_state: PQNState, 
        batch: ArrayTree,
    ) -> PQNState:
        """Update agent parameters with a batch of experience."""

        def td_error_loss(params: ArrayTree) -> Scalar:
            """Differentiable TD-error loss function."""
            
            # Predict Q-values for current observations
            state_q = agent_state.apply_fn(params, batch.observations)

            # Select Q-values for taken actions
            action_q = state_q[
                jnp.arange(self.rollout_steps)[:, None], 
                jnp.arange(self.num_envs)[None, :], 
                batch.actions
            ]

            # Compute TD-error loss as mean squared error
            return ((action_q - target_q) ** 2).mean()
        
        # Compute target Q-values using TD(λ)
        target_q = self.calculate_lambda_returns(agent_state, batch)

        # Compute TD-error loss and gradients
        loss, grads = jax.value_and_grad(td_error_loss)(agent_state.params)

        # Update model parameters with gradients
        agent_state = agent_state.apply_gradients(grads=grads)

        # Return updated agent state
        return agent_state

    def epsilon_decay(
        self,
        global_step: int
    ) -> Scalar:
        """Calculate current epsilon value."""
        
        decay_steps = self.epsilon_fraction * self.total_steps
        epsilon = self.epsilon_start + (self.epsilon_final - self.epsilon_start) * \
            (global_step / decay_steps)
        return jnp.maximum(epsilon, self.epsilon_final)

    @staticmethod
    def train(
        agent: 'PQN',
        seed: int = 0
    ) -> Dict:
        """Main training loop."""
        
        def train_step(carry: Dict, _: Any) -> Tuple[Dict, None]:
            """Scannable single train step."""

            # Unpack carry
            rng, agent_state, rollout_carry, global_step, logs = (
                carry['rng'], 
                carry['agent_state'], 
                carry['rollout_carry'], 
                carry['global_step'],
                carry['logs']
            )

            # Set current epsilon
            epsilon = agent.epsilon_decay(global_step)
            agent_state = agent_state.replace(epsilon=epsilon)

            # Generate experience batch
            rollout_result = agent.rollout(rollout_carry, agent_state)
            experiences, new_rollout_carry, rollout_logs = (
                rollout_result['experiences'],
                rollout_result['carry'],
                rollout_result['logs']
            )

            # Perform learning step
            agent_state = agent.learn(agent_state, experiences)

            # Update logs
            steps_per_rollout = agent.rollout_steps * agent.num_envs
            global_step = global_step + steps_per_rollout
            logs = Logs(
                rewards=logs.rewards.at[global_step // steps_per_rollout].set(rollout_logs.rewards),
                dones=logs.dones.at[global_step // steps_per_rollout].set(rollout_logs.dones),
                global_step=global_step
            )

            # Print logs if verbose
            checkpoint = jnp.argmax(agent.checkpoints == global_step) + 1
            jax.lax.cond(
                agent.verbose and jnp.any(agent.checkpoints == global_step),
                lambda: agent.print_logs(logs, checkpoint),
                lambda: None,
            )

            # Build carry for next step
            new_carry = {
                'rng': rng,
                'agent_state': agent_state,
                'rollout_carry': new_rollout_carry,
                'global_step': global_step,
                'logs': logs
            }

            return new_carry, None

        # Initialise RNG
        rng = jax.random.PRNGKey(seed)

        # Initialise train carry
        initial_carry = agent.init_train_carry(rng)

        # Scan the train step to train agent
        final_carry, _ = jax.lax.scan(
            f=train_step, init=initial_carry, xs=None, length=agent.num_rollouts
        )
        return final_carry
    