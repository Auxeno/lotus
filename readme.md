<div align="center">

  <h1> 🪷 Lotus</h1>
  
  <h3>A high-performance pure-JAX reinforcement learning library</h3>
  
  [![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

Lotus is a reinforcement learning library written in pure JAX. It supports `jit`, `vmap`, and `pmap` for efficient training and scaling across hardware accelerators.

### Example

Train multiple DQN agents on 100 different seeds in parallel:

```python
from lotus import DQN

# Create agent and seeds
agent = DQN.create(env='Breakout-MinAtar')
seeds = jnp.arange(100)

# Vectorised training
train_fn = jax.vmap(agent.train, in_axes=(None, 0))
trained_agents = train_fn(agent, seeds)
```

[![License](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/auxeno/lotus/blob/main/notebooks/demo.ipynb)

See the Colab notebook for more examples and advanced usage.