"""
Base Agent

Provides core functionality for agents.

Features:
- Agent creation
- Environment creation
- Environment step and reset API
- Checkpointing logic
- Log printing
- Evaluation
- Environment spaces
"""

from typing import Any, Optional, Tuple, Dict, Union, Sequence
import jax
import jax.numpy as jnp
from flax.struct import dataclass, field
from chex import Array, ArrayTree, PRNGKey
import gymnax

from .utils import Transition, AgentState, Logs
from .buffer import Buffer


@dataclass
class BaseAgent:

    env: Union[str, Any]       = field(False, default='CartPole-v1')
    env_params: ArrayTree      = field(False, default=None)
    num_checkpoints: int       = field(False, default=20)
    num_evalulations: int      = field(False, default=10)
    verbose: bool              = field(False, default=True)
    total_steps: int           = field(False, default=1_000_000)
    num_envs: int              = field(False, default=8)
    rollout_steps: int         = field(False, default=4)
    hidden_dims: Sequence[int] = field(False, default=(64, 64))
    anneal_lr: bool            = field(False, default=False)
    learning_rate: float       = field(True, default=3e-4)
    gamma: float               = field(True, default=0.99)
    max_grad_norm: float       = field(True, default=10.0)

    @classmethod
    def create(
        cls, 
        **kwargs
    ) -> 'BaseAgent':
        """Create an instance of BaseAgent."""

        # Create environment and add to kwargs
        env = kwargs.pop('env', cls.env)
        env_params = kwargs.pop('env_params', cls.env_params)
        env, env_params = cls.create_env(env, env_params)
        kwargs['env'] = env
        kwargs['env_params'] = env_params

        # Return instance of agent
        return cls(**kwargs)

    @staticmethod
    def create_env(
        env: Union[str, Any], 
        env_params: Optional[ArrayTree] = None
    ) -> Tuple[Any, ArrayTree]:
        """Create environment and parameters (Gymnax)."""

        if isinstance(env, str):
            env, default_params = gymnax.make(env)
        else:
            default_params = getattr(env, 'default_params', None)
        env_params = default_params if env_params is None else env_params
        
        return env, env_params

    def env_step(
        self, 
        key: PRNGKey, 
        env_states: ArrayTree, 
        actions: Array
    ) -> Dict:
        """Vectorised environment step (Gymnax API)."""

        # RNG
        keys = jax.random.split(key, self.num_envs)

        # Environment step
        next_observations, next_env_states, rewards, dones, infos = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(keys, env_states, actions, self.env_params)

        # Dummy truncations for Gymnax
        truncations = jnp.full_like(dones, False)

        return {
            'next_env_states': next_env_states,
            'next_observations': next_observations,
            'rewards': rewards,
            'terminations': dones,
            'truncations': truncations,
            'infos': infos
        }

    def env_reset(
        self,
        key: PRNGKey,
    ) -> Dict:
        """Vectorised environment reset (Gymnax API)."""

        # RNG
        keys = jax.random.split(key, self.num_envs)

        # Environment reset
        observations, env_states = jax.vmap(
            self.env.reset, in_axes=(0, None)
        )(keys, self.env_params)

        # Dummy info for Gymnax
        info = {}
        
        return {
            'env_states': env_states,
            'observations': observations,
            'info': info
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
            'carry': final_carry,
            'logs': logs
        }
    
    def evaluate(
        self
    ):
        """Evaluate agent's performance."""

        pass

    def print_logs(
        self,
        logs: Logs, 
        checkpoint: int,
        window: int = 100
    ) -> None:
        """Print recently logged metrics."""

        # Print header on first checkpoint
        jax.lax.cond(
            checkpoint == 1,
            lambda: jax.debug.print(
                "{header}", 
                header=(
                    f"{'Progress':>8}  |  "
                    f"{'Step':>11}  |  "
                    f"{'Episode':>9}  |  "
                    f"{'Mean Rew':>8}  |  "
                    f"{'Mean Len':>7}"
                )
            ),
            lambda: None
        )
        
        # Calculate metrics
        episodes = jnp.sum(logs.dones)
        progress = 100 * checkpoint / self.num_checkpoints
        
        # Calculate where to slice for means
        num_rollouts, rollout_steps, num_envs = logs.rewards.shape
        start = jnp.floor(((checkpoint - 1) / self.num_checkpoints) * num_rollouts).astype(jnp.int32)
        
        # Done episodes
        recent_episodes = jnp.sum(
            jax.lax.dynamic_slice_in_dim(logs.dones, start, window, axis=0)
        )
        recent_rewards = jnp.sum(
            jax.lax.dynamic_slice_in_dim(logs.rewards, start, window, axis=0)
        )

        # Calculate means
        mean_reward = jnp.where(
            episodes > 0, 
            recent_rewards / recent_episodes, 
            0.0
        )
        mean_length = jnp.where(
            recent_episodes > 0, 
            (window * rollout_steps * num_envs) / recent_episodes, 
            0.0
        )

        # Print formatted progress logs
        jax.debug.print(
            "{progress:>7.1f}%  |  "
            "{steps:>11,}  |  "
            "{episodes:>9,}  |  "
            "{mean_reward:>8.2f}  |  "
            "{mean_length:>8.1f}", 
            progress=progress, 
            episodes=episodes, 
            steps=logs.global_step,
            mean_reward=mean_reward,
            mean_length=mean_length
        )

    @property
    def observation_space(self):
        """Environment observation space."""
        return self.env.observation_space(self.env_params)

    @property
    def action_space(self):
        """Environment action space."""
        return self.env.action_space(self.env_params)

    @property
    def num_rollouts(self):
        """Total number of rollouts performed by agent."""
        return self.total_steps // (self.rollout_steps * self.num_envs)

    @property
    def checkpoints(self):
        """Training steps that are checkpoints."""
        steps_per_rollout = self.rollout_steps * self.num_envs
        checkpoints = ((jnp.arange(1, self.num_checkpoints + 1) * \
            self.num_rollouts // self.num_checkpoints) * steps_per_rollout
        )
        return checkpoints