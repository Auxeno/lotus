"""
SAC Agent

Soft actor critic agent.

Features:
- Learnable entropy regularisation coefficient
- Global grad norm clipping
- Vectorised environments
- Soft target network updates
"""

from typing import Any, Tuple, Dict, Sequence, Callable
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.struct import dataclass, field
from flax.linen.initializers import orthogonal
import optax
from distrax import Normal
from chex import Scalar, Array, ArrayTree, PRNGKey

from ..common.networks import MLP, SimpleCNN
from ..common.buffer import Buffer
from ..common.utils import AgentState, Logs
from .ddpg import DDPG


### Networks ###
    
class SoftActorNetwork(nn.Module):
    """SAC soft actor network stochastically predicts actions."""

    action_dim: int
    pixel_obs: bool
    hidden_dims: Sequence[int]
    action_scale: Array
    action_bias: Array

    @nn.compact
    def __call__(self, observations: Array, a_min: int=-5, a_max: int=2) -> Array:
        # Use CNN for pixel observations
        if self.pixel_obs:
            torso = SimpleCNN()
        else:
            torso = lambda x: x
        x = torso(observations)

        # MLP core
        x = MLP(self.hidden_dims)(x)

        x = nn.Dense(2 * self.action_dim, kernel_init=orthogonal(1.0))(x)
        mean, log_std = jnp.split(x, 2, axis=-1)

        # Transform action log sdev
        log_std = jax.nn.tanh(log_std)
        log_std = a_min + 0.5 * (a_max - a_min) * (log_std + 1.0)

        return mean, log_std

    def get_action(self, key: PRNGKey, observations: Array, a_min: int=-5, a_max: int=2) -> Array:
        mean, log_std = self.__call__(observations, a_min, a_max)

        # Sample action from distribution and transform
        dist = Normal(mean, jnp.exp(log_std))
        sampled_action = dist.sample(seed=key)
        tanh_action = jax.nn.tanh(sampled_action)
        action = self.action_scale * tanh_action + self.action_bias
        
        # Log prob of selecting action in transformed distribution
        log_prob = dist.log_prob(sampled_action)
        log_prob = log_prob - jnp.log(self.action_scale * (1.0 - tanh_action ** 2) + 1e-6)
        log_prob =jnp.sum(log_prob, axis=1)

        return action, log_prob


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
            variable_axes={'params': 0},
            split_rngs={'params': True},
            axis_size=self.num_critics
        )
        q_values = ensemble(self.pixel_obs, self.hidden_dims)(observations, actions)

        return q_values
    

class Alpha(nn.Module):
    """SAC entropy regularisation coefficient."""

    init_alpha: float = 1.0

    @nn.compact
    def __call__(self):
        log_alpha = self.param('log_alpha', lambda key: jnp.log(self.init_alpha))
        alpha = jnp.exp(log_alpha)
        return alpha
    
    
### Agent State ###

class ActorState(AgentState):
    """SAC actor state which has its own target params and optimiser."""

    action_fn: Callable = field(pytree_node=False)
    action_scale: Array = field(True)
    action_bias: Array = field(True)
    

class CriticState(AgentState):
    """SAC critic state which has its own target params and optimiser."""

    target_params: ArrayTree = field(True)


class AlphaState(AgentState):
    """Learnable entropy regularisation coefficient."""

    target_entropy: Scalar = field(pytree_node=True)


@dataclass
class SACState:
    """State of a SAC agent includes states of actor, critic and alpha."""

    actor: ActorState = field(True)
    critic: CriticState = field(True)
    alpha: AlphaState = field(True)
    

### Agent ###

@dataclass
class SAC(DDPG):
    """Soft actor-critic agent."""

    init_alpha: float    = field(True, default=0.5)       # SAC initial entropy term

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

        actor = SoftActorNetwork(action_dim, pixel_obs, self.hidden_dims, action_scale, action_bias)
        critic = CriticEnsemble(pixel_obs, self.hidden_dims)
        alpha = Alpha(self.init_alpha)

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
        return SACState(
            actor=ActorState.create(
                apply_fn=actor.apply,
                params=actor.init(key_actor, sample_obs),
                action_fn=lambda params, key, x: actor.apply(params, key, x, method=actor.get_action),
                action_scale=action_scale,
                action_bias=action_bias,
                tx=optimizer
            ),
            critic=CriticState.create(
                apply_fn=critic.apply,
                params=critic.init(key_critic, sample_obs[None, ...], sample_actions[None, ...]),
                target_params=critic.init(key_critic, sample_obs[None, ...], sample_actions[None, ...]),
                tx=optimizer
            ),
            alpha=AlphaState.create(
                apply_fn=alpha.apply,
                params=alpha.init(jax.random.PRNGKey(0)),
                target_entropy=-float(action_dim),
                tx=optimizer
            )
        )

    def select_action(
        self,
        key: PRNGKey,
        agent_state: SACState, 
        observations: Array
    ) -> Array:
        """Action selection logic."""

        # Forward pass through actor network to get actions
        actions, _ = agent_state.actor.action_fn(agent_state.actor.params, key, observations)
        return {'actions': actions}

    def learn(
            self,
            key: PRNGKey,
            agent_state: SACState, 
            batch: ArrayTree
        ) -> SACState:
            """Perform SAC learning update."""

            def critic_loss(params: ArrayTree) -> Scalar:
                """Differentiable critic loss function."""

                # Predict Q-values for current states and actions for ensemble
                action_q = agent_state.critic.apply_fn(
                    params, batch.observations, batch.actions
                )

                # Compute TD-error and mean squared error loss for ensemble
                return ((action_q - target_q[None, ...]) ** 2).mean()
            
            def actor_loss(params: ArrayTree) -> Scalar:
                """Differentiable actor loss function."""

                # Get actions from actor network for current observations
                actions, log_probs = agent_state.actor.action_fn(params, key_2, batch.observations)
                
                # Get Q-values from critic ensemble for the selected actions
                action_q = agent_state.critic.apply_fn(
                    agent_state.critic.params, batch.observations, actions
                )
                action_q = jnp.min(action_q, axis=0)

                # Entropy bonus
                entropy_bonus = agent_state.alpha.apply_fn(agent_state.alpha.params) * log_probs
                
                # Compute policy gradient loss (maximise Q-values)
                return (entropy_bonus - action_q).mean()
            
            def alpha_loss(params: ArrayTree) -> Scalar:
                """Differentiable alpha loss function."""

                alpha = agent_state.alpha.apply_fn(params)
                return (-alpha * (log_probs + agent_state.alpha.target_entropy)).mean()

            # RNG for 3 stochastic actor forward passes
            key_1, key_2, key_3 = jax.random.split(key, 3)

            # Compute target Q-values using target actor and critic networks
            next_actions, next_log_probs = agent_state.actor.action_fn(
                agent_state.actor.params, key_1, batch.next_observations
            )
            next_state_q = agent_state.critic.apply_fn(
                agent_state.critic.target_params, batch.next_observations, next_actions
            )

            # Penalty for high entropy actions
            entropy_penalty = agent_state.alpha.apply_fn(agent_state.alpha.params) * next_log_probs

            # Take minimum prediction from ensemble
            next_state_q = jnp.min(next_state_q, axis=0) - entropy_penalty

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

            # Compute alpha loss and gradients
            _, log_probs = agent_state.actor.action_fn(
                agent_state.actor.params, key_3, batch.observations
                )
            loss_alpha, grads_alpha = jax.value_and_grad(alpha_loss)(agent_state.alpha.params)

            # Update alpha parameters with gradients
            agent_state = agent_state.replace(
                alpha=agent_state.alpha.apply_gradients(grads=grads_alpha)
            )

            # Return updated agent state
            return agent_state
    
    @staticmethod
    def train(
        agent: 'SAC',
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
            rng, key_sample, key_learn = jax.random.split(rng, 3)

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
                    key_learn,
                    agent_state,
                    batch=Buffer.sample(key_sample, buffer_state, agent.batch_size)
                ),
                lambda: agent_state
            )

            # Soft target network update
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
