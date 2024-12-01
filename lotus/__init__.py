import warnings

from .algos import DQN


# Suppress warning caused by Gymnax
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*scatter inputs have incompatible types.*",
)