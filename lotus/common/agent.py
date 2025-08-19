"""
Base Agents

Provides core functionality for agents to inherit from.

BaseAgent features:
- Agent creation
- Environment creation
- Environment step and reset API
- Checkpointing logic
- Log printing
- Evaluation
- Environment spaces

OffPolicyAgent features:
- Train carry initialisation with replay buffer

OnPolicyAgent features:
- Train carry initialisation without replay buffer

RecurrentOnPolicyAgent features:
- Train carry initialisation with no replay buffer and hidden stat init
"""
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import gymnax
from chex import Array, ArrayTree, PRNGKey
from flax.struct import dataclass, field

from .buffer import Buffer
from .networks import GRUCore
from .utils import Transition, AgentState, Logs


@dataclass
class BaseAgent:
    env: str | Any = field(False, default="CartPole-v1")
    env_params: ArrayTree = field(False, default=None)
    num_checkpoints: int = field(False, default=20)
    num_evalulations: int = field(False, default=10)
    verbose: bool = field(False, default=True)
    total_steps: int = field(False, default=1_000_000)
    num_envs: int = field(False, default=8)
    rollout_steps: int = field(False, default=4)
    hidden_dims: Sequence[int] = field(False, default=(64, 64))
    anneal_lr: bool = field(False, default=False)
    learning_rate: float = field(True, default=3e-4)
    gamma: float = field(True, default=0.99)
    max_grad_norm: float = field(True, default=10.0)

    @classmethod
    def create(
        cls, 
        **kwargs
    ) -> "BaseAgent":
        """Create an instance of BaseAgent."""
        env = kwargs.pop("env", cls.env)
        env_params = kwargs.pop("env_params", cls.env_params)
        env, env_params = cls.create_env(env, env_params)
        kwargs["env"] = env
        kwargs["env_params"] = env_params
        return cls(**kwargs)

    @staticmethod
    def create_env(
        env: str | Any, 
        env_params: ArrayTree | None = None
    ) -> tuple[Any, ArrayTree | None]:
        """Create environment and parameters (Gymnax)."""
        if isinstance(env, str):
            env, default_params = gymnax.make(env)
        else:
            default_params = getattr(env, "default_params", None)
        env_params = default_params if env_params is None else env_params
        return env, env_params

    def env_step(
        self, 
        key: PRNGKey, 
        env_states: ArrayTree, 
        actions: Array
    ) -> dict:
        """Vectorised environment step (Gymnax API)."""
        keys = jax.random.split(key, self.num_envs)

        # Vectorised environment step
        next_observations, next_env_states, rewards, dones, infos = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(keys, env_states, actions, self.env_params)

        # Dummy truncations for Gymnax
        truncations = jnp.full_like(dones, False)

        return {
            "next_env_states": next_env_states,
            "next_observations": next_observations,
            "rewards": rewards,
            "terminations": dones,
            "truncations": truncations,
            "infos": infos
        }

    def env_reset(
        self,
        key: PRNGKey,
    ) -> dict:
        """Vectorised environment reset (Gymnax API)."""
        keys = jax.random.split(key, self.num_envs)

        # Vectorised environment reset
        observations, env_states = jax.vmap(
            self.env.reset, in_axes=(0, None)
        )(keys, self.env_params)

        # Dummy info for Gymnax
        info = {}
        
        return {
            "env_states": env_states,
            "observations": observations,
            "info": info
        }
    
    def rollout(
        self,
        initial_carry: dict,
        agent_state: AgentState,
    ) -> dict:
        """Collect experience from environment."""

        def rollout_step(carry: dict, _: Any) -> tuple[dict, tuple[Transition, Logs]]:
            """Scannable single vectorised environment step."""
            key, env_states, observations = (
                carry["key"], carry["env_states"], carry["observations"]
            )

            key, key_action, key_step = jax.random.split(key, 3)

            actions = self.select_action(
                key_action, agent_state, observations
            )["actions"]

            step_result = self.env_step(key_step, env_states, actions)

            # Build carry for next step
            new_carry = {
                "key": key,
                "env_states": step_result["next_env_states"],
                "observations": step_result["next_observations"]
            }

            # Build transition
            transition = Transition(
                observations=observations,
                next_observations=step_result["next_observations"],
                actions=actions,
                rewards=step_result["rewards"],
                terminations=step_result["terminations"],
                truncations=step_result["truncations"]
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
    
    def evaluate(
        self
    ) -> Any:
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
                    f"{"Progress":>8}  |  "
                    f"{"Step":>11}  |  "
                    f"{"Episode":>9}  |  "
                    f"{"Mean Rew":>8}  |  "
                    f"{"Mean Len":>7}"
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
    def params(self) -> dict:
        """
        Returns a dictionary of agent parameters.

        Each key is the parameter name, and each value is a tuple containing:
        - Parameter value
        - Parameter type
        - Whether the parameter supports vmapped operations
        """
        fields = self.__dataclass_fields__
        config = {
            field.name: (
                getattr(self, field.name, field.default),
                field.type,
                field.metadata.get("pytree_node")
            )
            for field in fields.values()
        }
        return config

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


@dataclass
class OffPolicyAgent(BaseAgent):
    def init_train_carry(
        self,
        rng: PRNGKey
    ) -> dict:
        """Set up the initial train carry."""
        rng, key_agent, key_reset, key_rollout = jax.random.split(rng, 4)
        dummy_key = jax.random.PRNGKey(0)
        
        # Initialise agent state
        agent_state = self.create_agent_state(key_agent)

        # Initialise buffer with a sample transition
        sample_transition = Transition(
            observations=self.observation_space.sample(dummy_key),
            next_observations=self.observation_space.sample(dummy_key),
            actions=self.action_space.sample(dummy_key),
            rewards=jnp.array(0.0, dtype=jnp.float32),
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
            "key": key_rollout,
            "env_states": reset_result["env_states"],
            "observations": reset_result["observations"]
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
            "rng": rng,
            "agent_state": agent_state,
            "buffer_state": buffer_state,
            "rollout_carry": rollout_carry,
            "global_step": 0,
            "logs": logs
        }
    

@dataclass
class OnPolicyAgent(BaseAgent):
    num_envs: int = field(False, default=8)
    rollout_steps: int = field(False, default=16)
    
    def init_train_carry(
        self,
        rng: PRNGKey
    ) -> dict:
        """Set up the initial train carry."""
        rng, key_agent, key_reset, key_rollout = jax.random.split(rng, 4)
        
        # Initialise agent state
        agent_state = self.create_agent_state(key_agent)

        # Initial observations and environment states
        reset_result = self.env_reset(key_reset)

        # Build initial rollout carry
        rollout_carry = {
            "key": key_rollout,
            "env_states": reset_result["env_states"],
            "observations": reset_result["observations"]
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
            "rng": rng,
            "agent_state": agent_state,
            "rollout_carry": rollout_carry,
            "global_step": 0,
            "logs": logs
        }
    
    
@dataclass
class RecurrentOnPolicyAgent(BaseAgent):
    num_envs: int = field(False, default=8)
    rollout_steps: int = field(False, default=16)
    
    def init_train_carry(
        self,
        rng: PRNGKey
    ) -> dict:
        """Set up the initial train carry."""
        rng, key_agent, key_reset, key_rollout = jax.random.split(rng, 4)
        
        # Initialise agent state
        agent_state = self.create_agent_state(key_agent)

        # Initial rnn hidden state
        initial_rnn_state = GRUCore.initialize_carry(self.num_envs, self.hidden_dims[-1])

        # Initial observations and environment states
        reset_result = self.env_reset(key_reset)

        # Initial dones
        initial_dones = jnp.zeros(self.num_envs, dtype=bool)

        # Build initial rollout carry
        rollout_carry = {
            "key": key_rollout,
            "env_states": reset_result["env_states"],
            "observations": reset_result["observations"],
            "rnn_state": initial_rnn_state,
            "prev_dones": initial_dones
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
            "rng": rng,
            "agent_state": agent_state,
            "rollout_carry": rollout_carry,
            "global_step": 0,
            "logs": logs
        }
    