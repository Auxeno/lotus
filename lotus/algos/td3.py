"""
TD3 Agent

Twin delayed DDPG agent.

Features:
- Global grad norm clipping
- Vectorised environments
- Soft target network updates
"""
from typing import Any, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import Scalar, Array, ArrayTree, PRNGKey
from flax.struct import dataclass, field
from flax.linen.initializers import orthogonal

from ..common.networks import MLP, SimpleCNN
from ..common.buffer import Buffer
from ..common.utils import AgentState, Logs
from .ddpg import DDPG, DDPGState, ActorState, CriticState


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
    
    
class CriticEnsemble(nn.Module):
    """Ensemble of critic networks."""
    pixel_obs: bool
    hidden_dims: Sequence[int]
    num_critics: int = 2

    @nn.compact
    def __call__(self, observations: Array, actions: Array) -> Array:
        ensemble = nn.vmap(
            target=CriticNetwork,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_critics
        )
        q_values = ensemble(self.pixel_obs, self.hidden_dims)(observations, actions)

        return q_values
    

### Agent State ###

# Alias
TD3State = DDPGState


### Agent ###

@dataclass
class TD3(DDPG):
    """Twin delayed DDPG agent."""
    actor_delay: int     = field(False, default=2)   # Critic train frequency
    noise_explore: float = field(True, default=0.1)  # Exploration noise sdev
    noise_policy: float  = field(True, default=0.2)  # Policy noise sdev
    noise_clip: float    = field(True, default=0.5)  # Noise threshold

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
        critic = CriticEnsemble(pixel_obs, self.hidden_dims)

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
        return TD3State(
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
    ) -> dict:
        """Action selection logic."""

        # Forward pass through actor network to get actions
        actions = agent_state.actor.apply_fn(agent_state.actor.params, observations)

        # Add noise to actions for exploration
        noise = jax.random.normal(key, actions.shape) * \
            self.noise_explore * agent_state.actor.action_scale

        return {
            "actions": jnp.clip(
                actions + noise, 
                -agent_state.actor.action_scale,
                agent_state.actor.action_scale
            ),
            "noise": noise
        }

    def learn(
        self,
        key: PRNGKey,
        agent_state: AgentState,
        batch: ArrayTree,
        train_actor: bool
    ) -> AgentState:
        """Update agent parameters with a batch of experience."""
        
        def critic_loss(params: ArrayTree) -> Scalar:
            """Differentiable critic loss function."""

            # Q-values for current observations and actions
            action_q = agent_state.critic.apply_fn(
                params, batch.observations, batch.actions
            )
            
            # Compute TD-error and mean squared error
            return ((action_q - target_q[None, ...]) ** 2).mean()
        
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

        # Add clipped noise to actions
        noise = jax.random.normal(key, batch.actions.shape) * \
            self.noise_policy * agent_state.actor.action_scale
        noise_clipped = jnp.clip(noise, -self.noise_clip, self.noise_clip)
        next_actions = jnp.clip(
            next_actions + noise_clipped,
            -agent_state.actor.action_scale,
            agent_state.actor.action_scale,
        )

        # Take minimum prediction from ensemble
        next_state_q = agent_state.critic.apply_fn(
            agent_state.critic.target_params, batch.next_observations, next_actions
        ).min(axis=0)

        # Bellman equation for target Q-values
        target_q = batch.rewards + self.gamma * (1.0 - batch.terminations) * next_state_q

        # Compute critic loss and its gradients
        loss_critic, grads_critic = jax.value_and_grad(critic_loss)(agent_state.critic.params)

        # Update critic parameters with gradients
        agent_state = agent_state.replace(
            critic=agent_state.critic.apply_gradients(grads=grads_critic)
        )

        def actor_learn(agent_state):
            # Compute actor loss and its gradients
            loss_actor, grads_actor = jax.value_and_grad(actor_loss)(agent_state.actor.params)

            # Update actor parameters with gradients
            return agent_state.replace(
                actor=agent_state.actor.apply_gradients(grads=grads_actor)
            )

        # Delayed actor update
        agent_state = jax.lax.cond(
            train_actor,
            lambda x: actor_learn(x),
            lambda x: x,
            agent_state
        )

        # Return updated agent state
        return agent_state
    
    @staticmethod
    def train(
        agent: "TD3",
        seed: int = 0
    ) -> dict:
        """Main training loop."""
        
        def train_step(carry: dict, _: Any) -> tuple[dict, None]:
            """Scannable single train step."""

            # Unpack carry
            rng, agent_state, buffer_state, rollout_carry, global_step, logs = (
                carry["rng"], 
                carry["agent_state"], 
                carry["buffer_state"], 
                carry["rollout_carry"], 
                carry["global_step"],
                carry["logs"]
            )

            # RNG
            rng, key_sample, key_learn = jax.random.split(rng, 3)

            # Generate experience batch
            rollout_result = agent.rollout(rollout_carry, agent_state)
            experiences, new_rollout_carry, rollout_logs = (
                rollout_result["experiences"],
                rollout_result["carry"],
                rollout_result["logs"]
            )

            # Store experiences in buffer
            buffer_state, _ = jax.lax.scan(
                lambda buffer_state, experience: (Buffer.push(buffer_state, experience), None), 
                init=buffer_state, 
                xs=experiences
            )

            # Perform learn step
            train_actor = global_step // (agent.rollout_steps * agent.num_envs) % agent.actor_delay == 0
            agent_state = jax.lax.cond(
                buffer_state.size >= max(agent.batch_size, agent.learning_starts),
                lambda: agent.learn(
                    key_learn,
                    agent_state,
                    batch=Buffer.sample(key_sample, buffer_state, agent.batch_size),
                    train_actor=train_actor
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
                "rng": rng,
                "agent_state": agent_state,
                "buffer_state": buffer_state,
                "rollout_carry": new_rollout_carry,
                "global_step": global_step,
                "logs": logs
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
    