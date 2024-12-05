"""
RPPO Agent

Recurrent PPO agent.
Does not support environment timeouts via truncation.

Features:
- GRU core in network
- Reduced output layer variance
- GAE
"""

from typing import Any, Tuple, Dict, Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.struct import dataclass, field
from flax.linen.initializers import orthogonal
import optax
from distrax import Categorical
from chex import Scalar, Array, ArrayTree, PRNGKey

from ..common.agent import RecurrentOnPolicyAgent
from ..common.networks import MLP, SimpleCNN, GRUCore
from ..common.utils import AgentState, Transition, Logs


### Network ###
    
class RecurrentActorCriticNetwork(nn.Module):
    """Combined actor critic networks."""

    action_dim: int
    pixel_obs: bool
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, rnn_state: Array, observations: Array, resets: Array):
        # Use CNN for pixel observations
        if self.pixel_obs:
            torso = SimpleCNN(activation_fn=nn.relu)
        else:
            torso = lambda x: x
        x = torso(observations)

        # Apply MLP to torso output
        x = MLP(self.hidden_dims, activation_fn=nn.relu)(x)

        # GRU core
        new_rnn_state, x = GRUCore()(rnn_state, (x, resets))

        # Separate actor and critic heads
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)

        return new_rnn_state, logits, value.squeeze(-1)


### Agent State ###

# Alias
RPPOState = AgentState


### Environment Transition ###

@dataclass
class RPPOTransition:
    """Extended transition for better efficiency. Combined dones flag."""

    observations: Array = field(True)
    next_observations: Array = field(True)
    actions: Array = field(True)
    rewards: Array = field(True)
    dones: Array = field(True)
    log_probs: Array = field(True)
    values: Array = field(True)
    prev_dones: Array = field(True)
    initial_rnn_state: Array = field(True, default=jnp.nan)
    final_value: Array = field(True, default=jnp.nan)


### Agent ###

@dataclass
class RPPO(RecurrentOnPolicyAgent):
    """Recurrent PPO agent."""

    num_epochs: int      = field(False, default=10)   # Number of training epochs per rollout
    num_minibatches: int = field(False, default=1)    # Number of minibatches per epoch
    gae_lambda: float    = field(True, default=0.95)  # GAE lambda for advantage estimation
    clip_coef: float     = field(True, default=0.2)   # PPO clipping coefficient
    advantage_norm: bool = field(True, default=True)  # Normalise advantages
    entropy_bonus: float = field(True, default=0.01)  # Entropy bonus for exploration
    value_weight: float  = field(True, default=0.5)   # Weight for value loss

    def create_agent_state(
        self,
        key: PRNGKey
    ) -> AgentState:
        """Initialise network, parameters and optimiser."""

        # Create network
        action_dim = self.action_space.n
        obs_shape = self.observation_space.shape
        sample_obs = self.observation_space.sample(jax.random.PRNGKey(0))[None, None, ...]
        sample_resets = jnp.zeros((1, self.num_envs), dtype=bool)
        sample_hidden = GRUCore.initialize_carry(self.num_envs, self.hidden_dims[-1])
        if len(obs_shape) not in (1, 3):
            raise Exception(f"Invalid observation space shape: {obs_shape}.")
        pixel_obs = len(obs_shape) == 3
        network = RecurrentActorCriticNetwork(
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
        return RPPOState.create(
            apply_fn=network.apply,
            params=network.init(key, sample_hidden, sample_obs, sample_resets),
            tx=optimizer
        )
    
    def select_action(
        self,
        key: PRNGKey, 
        agent_state: RPPOState,
        rnn_state: Array,
        observations: Array,
        dones: Array
    ) -> Array:
        """Select action using recurrent policy network."""

        # Forward pass through network, None adds a dummy timestep dimension
        new_rnn_state, logits, values = agent_state.apply_fn(
            agent_state.params, rnn_state, observations[None, ...], dones[None, ...]
        )
        
        # Create categorical distribution
        dist = Categorical(logits=logits.squeeze(0))
        
        # Sample action from policy distribution
        actions = dist.sample(seed=key)
        
        # Calculate log probs
        log_probs = dist.log_prob(actions)

        return {
            'actions': actions, 
            'new_rnn_state': new_rnn_state, 
            'log_probs': log_probs, 
            'values': values.squeeze(0)
        }
    
    def rollout(
        self,
        initial_carry: Dict,
        agent_state: AgentState,
    ):
        """Collect experience from environment."""
        
        def rollout_step(carry: Dict, _: Any) -> Tuple[Dict, Transition]:
            """Scannable single vectorised environment step."""

            # Unpack carry
            key, env_states, observations, rnn_state, prev_dones = (
                carry['key'], 
                carry['env_states'], 
                carry['observations'],
                carry['rnn_state'],
                carry['prev_dones']
            )

            # RNG
            key, key_action, key_step = jax.random.split(key, 3)

            # Action selection
            action_result = self.select_action(
                key_action, agent_state, rnn_state, observations, prev_dones
            )

            # Environment step
            step_result = self.env_step(key_step, env_states, action_result['actions'])

            # Calculate dones
            dones = jnp.logical_or(step_result['terminations'], step_result['truncations'])

            # Build carry for next step
            new_carry = {
                'key': key,
                'env_states': step_result['next_env_states'],
                'observations': step_result['next_observations'],
                'rnn_state': action_result['new_rnn_state'],
                'prev_dones': dones
            }

            # Build transition
            transition = RPPOTransition(
                observations=observations,
                next_observations=step_result['next_observations'],
                actions=action_result['actions'],
                rewards=step_result['rewards'],
                dones=dones,
                log_probs=action_result['log_probs'],
                values=action_result['values'],
                prev_dones=prev_dones
            )

            # Build logs for step
            logs = Logs(rewards=step_result['rewards'], dones=dones)

            return new_carry, (transition, logs)
            
        # Scan to generate a batch of transitions
        final_carry, (experiences, logs) = jax.lax.scan(
            f=rollout_step, init=initial_carry, xs=None, length=self.rollout_steps
        )

        # Add initial RNN state to batch of experiences
        experiences = experiences.replace(initial_rnn_state=initial_carry['rnn_state'])

        # Return experiences, logs and final carry
        return {
            'experiences': experiences,
            'carry': final_carry,
            'logs': logs
        }

    def calculate_gae(
        self,
        batch: ArrayTree,
    ) -> Tuple:
        """Compute advantage and returns using GAE."""

        def gae_step(advantage, transition) -> Tuple[Array, ArrayTree]:
            """Scannable GAE step."""
            # Unpack transition
            reward, done, value, next_value = transition

            # Mask for non-terminal transitions
            non_terminal = 1.0 - done

            # Compute delta (TD residual)
            delta = reward + self.gamma * next_value * non_terminal - value

            # Update advantage
            advantage = delta + self.gamma * self.gae_lambda * non_terminal * advantage
            return advantage, advantage

        # Values for next observations
        next_values = jnp.concat((batch.values[1:], batch.final_value[None, ...]), axis=0)
        
        # Initialise GAE scan parameters
        initial_carry = jnp.zeros(self.num_envs)
        transitions = (batch.rewards, batch.dones, batch.values, next_values)

        # Compute advantages via reversed scan
        _, advantages = jax.lax.scan(
            gae_step,
            initial_carry,
            transitions,
            reverse=True
        )

        # Calculate returns
        returns = advantages + batch.values

        return advantages, returns

    def learn(
        self,
        key: PRNGKey, 
        agent_state: RPPOState, 
        batch: ArrayTree, 
    ) -> RPPOState:
        """Update agent parameters with a batch of experience."""
        
        def minibatch_update(agent_state: RPPOState, mb_indices: Array) -> RPPOState:
            """Scannable minibatch gradient descent update."""
            
            def ppo_loss(params: ArrayTree) -> Scalar:
                """Differentiable PPO loss function."""

                # Full forward pass
                _, logits, values = agent_state.apply_fn(
                    params, batch.initial_rnn_state, batch.observations, batch.prev_dones
                )
                
                # Flatten batch, logits and values
                actions, old_log_probs, logits, values = \
                    jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), (batch.actions, batch.log_probs, logits, values))

                # Policy loss
                distribution = Categorical(logits=logits[mb_indices])
                log_probs = distribution.log_prob(actions[mb_indices])
                ratio = jnp.exp(log_probs - old_log_probs[mb_indices])
                loss_surrogate_unclipped = -advantages[mb_indices] * ratio
                loss_surrogate_clipped = -advantages[mb_indices] * \
                    jnp.clip(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                loss_policy = jnp.maximum(loss_surrogate_unclipped, loss_surrogate_clipped).mean()

                # Value loss
                loss_value = ((values[mb_indices] - returns[mb_indices]) ** 2).mean()

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
        advantages, returns = self.calculate_gae(batch)

        # Normalise advantages if enabled            
        advantages = jnp.where(
            self.advantage_norm,
            (advantages - advantages.mean()) / (advantages.std() + 1e-8),
            advantages
        )

        # Flatten advantages and returns
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
        agent: 'RPPO',
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

            # RNG
            rng, key_learn = jax.random.split(rng)

            # Generate experience batch
            rollout_result = agent.rollout(rollout_carry, agent_state)
            experiences, new_rollout_carry, rollout_logs = (
                rollout_result['experiences'],
                rollout_result['carry'],
                rollout_result['logs']
            )

            # Unpack items rollout carry
            new_rnn_state, next_observations, next_dones = (
                new_rollout_carry['rnn_state'],
                new_rollout_carry['observations'],
                new_rollout_carry['prev_dones']
            )

            # Add final value to batch
            _, _, final_value = agent_state.apply_fn(
                agent_state.params, new_rnn_state, next_observations[None, ...], next_dones[None, ...]
            )
            experiences = experiences.replace(final_value=final_value.squeeze(0))

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
    