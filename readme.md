<div align="center">

  <h1> 🪷 Lotus</h1>
  
  <h3>A high-performance pure-JAX reinforcement learning library</h3>
  
  [![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

### Description

Lotus is a reinforcement learning library written in pure JAX. It supports `jit`, `vmap`, and `pmap` for efficient training and scaling across hardware accelerators.

### Example

Train multiple DQN agents on 100 different seeds in parallel:

```python
from lotus import DQN

# Create agent and seeds
agent = DQN.create(env='Breakout-MinAtar')
seeds = jnp.arange(100)

trained_agents = jax.vmap(agent.train, in_axes=(None, 0))(agent, seeds)
```

[![License](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Auxeno/lotus/tree/main)

See the Colab notebook for more examples and advanced usage.