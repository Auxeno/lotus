"""
DDPG Agent

Deep deterministic policy gradient agent.

Features:
- Gaussian noise instead of OU noise
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

from ..common.agent import OffPolicyAgent
from ..common.networks import MLP, SimpleCNN
from ..common.buffer import Buffer
from ..common.utils import AgentState, Logs


### Networks ###
    
class ActorNetwork(nn.Module):
    """DDPG actor network outputs continuous actions."""

    action_dim: int
    pixel_obs: bool
    hidden_dims: Sequence[int]
    action_scale: Array
    action_bias: Array

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

        # Tanh transformed and scaled actionsactions
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0))(x)
        x = jnp.tanh(x)
        action = x * self.action_scale + self.action_bias

        return action
    

class CriticNetwork(nn.Module):
    """DDPG critic with configurable torso."""

    pixel_obs: bool
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: Array, actions: Array) -> Array:
        # Use CNN for pixel observations
        if self.pixel_obs:
            torso = SimpleCNN()
        else:
            torso = lambda x: x
        x = torso(observations)

        # Concatenate latent vector with actions
        x = jnp.concatenate([x, actions], axis=-1)

        # MLP core
        x = MLP(self.hidden_dims)(x)

        q_values = nn.Dense(1, kernel_init=orthogonal(1.0))(x)

        return q_values.squeeze(-1)
    
    
### Agent State ###

class ActorState(AgentState):
    """DDPG actor state which has its own target params and optimiser."""

    target_params: ArrayTree = field(True)
    action_scale: Array = field(True)
    action_bias: Array = field(True)
    

class CriticState(AgentState):
    """DDPG critic state which has its own target params and optimiser."""

    target_params: ArrayTree = field(True)


@dataclass
class DDPGState:
    """State of a DDPG agent includes states of actor and critic."""

    actor: ActorState = field(True)
    critic: CriticState = field(True)


### Agent ###

@dataclass
class DDPG(OffPolicyAgent):
    """Deep deterministic policy gradient agent."""

    batch_size: int         = field(False, default=64)       # Replay buffer sample size
    learning_starts: int    = field(False, default=1000)     # Begin learning after
    buffer_capacity: int    = field(False, default=100_000)  # Replay buffer capacity
    tau: float              = field(True, default=0.05)      # Soft target update tau
    noise_sigma: float      = field(True, default=0.1)       # Gaussian noise sdev

    def create_agent_state(
        self,
        key: PRNGKey
    ) -> AgentState:
        """Initialise network, parameters and optimiser."""

        # RNG
        key_actor, key_critic = jax.random.split(key)

        # Create network
        action_dim = jnp.prod(*self.action_space.shape).item()
        action_scale = jnp.array((self.action_space.high - self.action_space.low) / 2)
        action_bias = jnp.array((self.action_space.high + self.action_space.low) / 2)
        obs_shape = self.observation_space.shape
        sample_obs = self.observation_space.sample(jax.random.PRNGKey(0))
        sample_actions = self.action_space.sample(jax.random.PRNGKey(0))
        
        if len(obs_shape) not in (1, 3):
            raise Exception(f"Invalid observation space shape: {obs_shape}.")
        pixel_obs = len(obs_shape) == 3

        actor = ActorNetwork(action_dim, pixel_obs, self.hidden_dims, action_scale, action_bias)
        critic = CriticNetwork(pixel_obs, self.hidden_dims)
        
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
        return DDPGState(
            actor=ActorState.create(
                apply_fn=actor.apply,
                params=actor.init(key_actor, sample_obs[None, ...]),
                target_params=actor.init(key_actor, sample_obs[None, ...]),
                action_scale=action_scale,
                action_bias=action_bias,
                tx=optimizer
            ),
            critic=CriticState.create(
                apply_fn=critic.apply,
                params=critic.init(key_critic, sample_obs[None, ...], sample_actions[None, ...]),
                target_params=critic.init(key_critic, sample_obs[None, ...], sample_actions[None, ...]),
                tx=optimizer
            )
        )

    def select_action(
        self,
        key: PRNGKey,
        agent_state: AgentState,
        observations: Array
    ):
        """Action selection logic."""

        # Forward pass through actor network to get actions
        actions = agent_state.actor.apply_fn(agent_state.actor.params, observations)

        # Add noise to actions for exploration
        noise = jax.random.normal(key, actions.shape) * \
            self.noise_sigma * agent_state.actor.action_scale

        return {
            'actions': jnp.clip(
                actions + noise, 
                -agent_state.actor.action_scale,
                agent_state.actor.action_scale
            ),
            'noise': noise
        }

    def learn(
        self,
        agent_state: AgentState,
        batch: ArrayTree
    ) -> AgentState:
        """Update agent parameters with a batch of experience."""
        
        def critic_loss(params: ArrayTree) -> Scalar:
            """Differentiable critic loss function."""

            # Q-values for current observations and actions
            action_q = agent_state.critic.apply_fn(
                params, batch.observations, batch.actions
            )
            
            # Compute TD-error and mean squared error
            return ((action_q - target_q) ** 2).mean()
        
        def actor_loss(params: ArrayTree) -> Scalar:
            """Differentiable actor loss function."""

            # Get actions from actor network for current observations
            actions = agent_state.actor.apply_fn(params, batch.observations)
            
            # Get Q-values from critic network for the selected actions
            action_q = agent_state.critic.apply_fn(
                agent_state.critic.params, batch.observations, actions
            )
            
            # Compute policy gradient loss (maximise Q-values)
            return -action_q.mean()

        # Compute target Q-values using target actor and critic networks
        next_actions = agent_state.actor.apply_fn(
            agent_state.actor.target_params, batch.next_observations
        )
        next_state_q = agent_state.critic.apply_fn(
            agent_state.critic.target_params, batch.next_observations, next_actions
        )

        # Bellman equation for target Q-values
        target_q = batch.rewards + self.gamma * (1.0 - batch.terminations) * next_state_q

        # Compute critic loss and its gradients
        loss_critic, grads_critic = jax.value_and_grad(critic_loss)(agent_state.critic.params)

        # Update critic parameters with gradients
        agent_state = agent_state.replace(
            critic=agent_state.critic.apply_gradients(grads=grads_critic)
        )

        # Compute actor loss and its gradients
        loss_actor, grads_actor = jax.value_and_grad(actor_loss)(agent_state.actor.params)

        # Update actor parameters with gradients
        agent_state = agent_state.replace(
            actor=agent_state.actor.apply_gradients(grads=grads_actor)
        )

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

    @staticmethod
    def train(
        agent: 'DDPG',
        seed: int = 0
    ) -> Dict:
        """Main training loop."""
        
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
            rng, key_sample = jax.random.split(rng)

            # Generate experience batch
            rollout_result = agent.rollout(rollout_carry, agent_state)
            experiences, new_rollout_carry, rollout_logs = (
                rollout_result['experiences'],
                rollout_result['carry'],
                rollout_result['logs']
            )

            # Store experiences in buffer
            buffer_state, _ = jax.lax.scan(
                lambda buffer_state, experience: (Buffer.push(buffer_state, experience), None), 
                init=buffer_state, 
                xs=experiences
            )

            # Perform learn step
            agent_state = jax.lax.cond(
                buffer_state.size >= max(agent.batch_size, agent.learning_starts),
                lambda: agent.learn(
                    agent_state,
                    batch=Buffer.sample(key_sample, buffer_state, agent.batch_size)
                ),
                lambda: agent_state
            )

            # Soft target network update
            new_target_actor_params = agent.soft_update(
                agent_state.actor.params, agent_state.actor.target_params
            )
            agent_state = agent_state.replace(
                actor=agent_state.actor.replace(target_params=new_target_actor_params)
            )
            new_target_critic_params = agent.soft_update(
                agent_state.critic.params, agent_state.critic.target_params
            )
            agent_state = agent_state.replace(
                critic=agent_state.critic.replace(target_params=new_target_critic_params)
            )

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
                'buffer_state': buffer_state,
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
    