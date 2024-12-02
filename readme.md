<div align="center">

  <h1 style="font-size: 3.5rem;"> 🪷 Lotus</h1>
  
  <h3>A high-performance JAX reinforcement learning library</h3>
  
  [![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

Lotus is a lightweight reinforcement learning library written in pure JAX (Flax). It supports `jit`, `vmap`, and `pmap` for fast and scalable training on hardware accelerators.

### Install

Clone the repository and install dependencies:

```
git clone https://github.com/auxeno/lotus
pip install -r lotus/requirements.txt
```

### Quick Start

Train multiple PPO agents on 100 different seeds in parallel:

```python
from lotus import PPO

# Create agent and seeds
agent = PPO.create(env='Breakout-MinAtar')
seeds = jnp.arange(100)

# Vectorised training
train_fn = jax.vmap(agent.train, in_axes=(None, 0))
trained_agents = train_fn(agent, seeds)
```

[![License](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/auxeno/lotus/blob/main/notebooks/demo.ipynb)

See the Colab notebook for more examples and advanced usage.

### Supported Algorithms

| Algorithm     | Discrete | Continuous | Paper                        |
|---------------|----------|------------|------------------------------|
| [DQN](https://github.com/Auxeno/lotus/blob/main/lotus/algos/dqn.py)           | ✔        | ✘          | [Mnih et al. 2013](https://arxiv.org/abs/1312.5602) |
| [QR-DQN](https://github.com/Auxeno/lotus/blob/main/lotus/algos/qrdqn.py)        | ✔        | ✘          | [Dabney et al. 2017](https://arxiv.org/abs/1710.10044) |
| [PPO](https://github.com/Auxeno/lotus/blob/main/lotus/algos/ppo.py)           | ✔        | ✘          | [Schulman et al. 2017](https://arxiv.org/abs/1707.06347) |
