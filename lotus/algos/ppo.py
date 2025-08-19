"""
PPO Agent

Proximal policy optimisation agent.

Features:
- Reduced output layer variance
- GAE
- Tanh network activation
"""
from typing import Any, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import Scalar, Array, ArrayTree, PRNGKey
from distrax import Categorical
from flax.struct import dataclass, field
from flax.linen.initializers import orthogonal

from ..common.agent import OnPolicyAgent
from ..common.networks import MLP, SimpleCNN
from ..common.utils import AgentState, Transition, Logs


### Network ###
    
class ActorCriticNetwork(nn.Module):
    """Combined actor critic networks."""
    action_dim: int
    pixel_obs: bool
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: Array):
        # Use CNN for pixel observations
        if self.pixel_obs:
            torso = SimpleCNN(activation_fn=nn.tanh)
        else:
            torso = lambda x: x
        x = torso(observations)

        # MLP core
        x = MLP(self.hidden_dims, activation_fn=nn.tanh)(x)

        # Separate actor and critic heads
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)

        return logits, value.squeeze(-1)
    

### Agent State ###

# Alias
PPOState = AgentState


### Environment Transition ###

@dataclass
class PPOTransition(Transition):
    """Extended transition for better efficiency."""
    log_probs: Array = field(True)
    values: Array = field(True)


### Agent ###

@dataclass
class PPO(OnPolicyAgent):
    """PPO agent."""
    num_epochs: int = field(False, default=10)        # Number of training epochs per rollout
    num_minibatches: int = field(False, default=1)    # Number of minibatches per epoch
    gae_lambda: float = field(True, default=0.95)     # GAE lambda for advantage estimation
    clip_coef: float = field(True, default=0.2)       # PPO clipping coefficient
    advantage_norm: bool = field(True, default=True)  # Normalise advantages
    entropy_bonus: float = field(True, default=0.01)  # Entropy bonus for exploration
    value_weight: float = field(True, default=0.5)    # Weight for value loss

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
        network = ActorCriticNetwork(
            action_dim, pixel_obs, self.hidden_dims
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
        return PPOState.create(
            apply_fn=network.apply,
            params=network.init(key, sample_obs[None, ...]),
            tx=optimizer
        )
    
    def select_action(
        self, 
        key: PRNGKey, 
        agent_state: AgentState,
        observations: Array
    ) -> dict:
        """Action selection logic."""
        # Forward pass through network
        logits, values = agent_state.apply_fn(agent_state.params, observations)
        
        # Create categorical distribution
        dist = Categorical(logits=logits)
        
        # Sample action from policy distribution
        actions = dist.sample(seed=key)
        
        # Calculate log probs
        log_probs = dist.log_prob(actions)
        
        return {"actions": actions, "log_probs": log_probs, "values": values}
    
    def rollout(
        self,
        initial_carry: dict,
        agent_state: AgentState,
    ) -> dict:
        """Collect experience from environment."""
        
        def rollout_step(carry: dict, _: Any) -> tuple[dict, Transition]:
            """Scannable single vectorised environment step."""

            # Unpack carry
            key, env_states, observations = (
                carry["key"], carry["env_states"], carry["observations"]
            )

            # RNG
            key, key_action, key_step = jax.random.split(key, 3)

            # Action selection
            action_result = self.select_action(key_action, agent_state, observations)
            actions = action_result["actions"]

            # Environment step
            step_result = self.env_step(key_step, env_states, actions)

            # Build carry for next step
            new_carry = {
                "key": key,
                "env_states": step_result["next_env_states"],
                "observations": step_result["next_observations"]
            }

            # Build transition
            transition = PPOTransition(
                observations=observations,
                next_observations=step_result["next_observations"],
                actions=actions,
                rewards=step_result["rewards"],
                terminations=step_result["terminations"],
                truncations=step_result["truncations"],
                log_probs=action_result["log_probs"],
                values=action_result["values"]
            )

            # Build logs for step
            dones = jnp.logical_or(step_result["terminations"], step_result["truncations"])
            logs = Logs(rewards=step_result["rewards"], dones=dones)

            return new_carry, (transition, logs)
            
        # Scan to generate a batch of transitions
        final_carry, (experiences, logs) = jax.lax.scan(
            f=rollout_step, init=initial_carry, xs=None, length=self.rollout_steps
        )

        return {
            "experiences": experiences,
            "carry": final_carry,
            "logs": logs
        }
    
    def calculate_gae(
        self,
        agent_state: AgentState, 
        batch: ArrayTree
    ) -> tuple:
        """Compute advantage and returns using GAE."""

        def gae_step(advantage, transition) -> tuple[Array, ArrayTree]:
            """Scannable GAE step."""
            reward, termination, truncation, value, next_value = transition

            # Mask for non-terminal transitions
            non_termination, non_truncation = 1.0 - termination, 1.0 - truncation

            # Compute delta (TD residual)
            delta = reward + self.gamma * next_value * non_termination - value

            # Update advantage
            advantage = delta + self.gamma * self.gae_lambda * non_termination * non_truncation * advantage
            return advantage, advantage

        # Values for current and next observations
        _, next_values = agent_state.apply_fn(agent_state.params, batch.next_observations)

        # Initialise GAE scan parameters
        initial_carry = jnp.zeros(self.num_envs)
        transitions = (batch.rewards, batch.terminations, batch.truncations, batch.values, next_values)

        # Compute advantages via reversed scan
        _, advantages = jax.lax.scan(
            f=gae_step, init=initial_carry, xs=transitions, reverse=True
        )

        returns = advantages + batch.values

        return advantages, returns

    def learn(
        self,
        key: PRNGKey, 
        agent_state: AgentState, 
        batch: ArrayTree
    ) -> AgentState:
        """Update agent parameters with a batch of experience."""
        
        def minibatch_update(agent_state: AgentState, mb_indices: Array) -> AgentState:
            """Scannable minibatch gradient descent update."""
            
            def ppo_loss(params: ArrayTree) -> Scalar:
                """Differentiable PPO loss function."""
                
                # Forward pass
                logits, values = agent_state.apply_fn(params, batch.observations[mb_indices])
                
                # Policy loss
                distribution = Categorical(logits=logits)
                log_probs = distribution.log_prob(batch.actions[mb_indices])
                ratio = jnp.exp(log_probs - batch.log_probs[mb_indices])
                loss_surrogate_unclipped = -advantages[mb_indices] * ratio
                loss_surrogate_clipped = -advantages[mb_indices] * \
                    jnp.clip(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                loss_policy = jnp.maximum(loss_surrogate_unclipped, loss_surrogate_clipped).mean()

                # Value loss
                loss_value = ((values - returns[mb_indices]) ** 2).mean()

                # Entropy bonus
                loss_entropy = distribution.entropy().mean()

                # Combine losses
                loss = (
                    loss_policy +
                    self.value_weight * loss_value +
                   -self.entropy_bonus * loss_entropy
                )
                return loss

            # Calculate PPO loss and gradients
            loss, grads = jax.value_and_grad(ppo_loss)(agent_state.params)

            # Update model parameters
            agent_state = agent_state.apply_gradients(grads=grads)

            return agent_state, loss

        # Compute advantages using GAE
        advantages, returns = self.calculate_gae(agent_state, batch)

        # Normalise advantages if enabled            
        advantages = jnp.where(
            self.advantage_norm,
            (advantages - advantages.mean()) / (advantages.std() + 1e-8),
            advantages
        )

        # Flatten data
        batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), batch)
        advantages, returns = jax.tree.map(lambda x: x.flatten(), (advantages, returns))

        # Create shuffled minibatch indices
        batch_size = self.rollout_steps * self.num_envs
        indices = jnp.tile(jnp.arange(batch_size), (self.num_epochs, 1))
        indices = jax.vmap(jax.random.permutation)(jax.random.split(key, self.num_epochs), indices)
        indices = indices.reshape(self.num_epochs * self.num_minibatches, -1)

        # Scan over minibatch indices for updates
        agent_state, losses = jax.lax.scan(
            f=minibatch_update, init=agent_state, xs=indices
        )
        
        # Return updated agent state
        return agent_state

    @staticmethod
    def train(
        agent: "PPO",
        seed: int = 0
    ) -> dict:
        """Main training loop."""
        
        def train_step(carry: dict, _: Any) -> tuple[dict, None]:
            """Scannable single train step."""

            # Unpack carry
            rng, agent_state, rollout_carry, global_step, logs = (
                carry["rng"], 
                carry["agent_state"],
                carry["rollout_carry"], 
                carry["global_step"],
                carry["logs"]
            )

            # RNG
            rng, key_learn = jax.random.split(rng)

            # Generate experience batch
            rollout_result = agent.rollout(rollout_carry, agent_state)
            experiences, new_rollout_carry, rollout_logs = (
                rollout_result["experiences"],
                rollout_result["carry"],
                rollout_result["logs"]
            )

            # Perform learn step
            agent_state = agent.learn(key_learn, agent_state, experiences)

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
    