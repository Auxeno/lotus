import numpy as np
import matplotlib.pyplot as plt


def process_logs(logs):
    """Processes the logs to extract episodic rewards and their corresponding steps."""

    # Get dones and rewards
    dones = logs.dones
    rewards = logs.rewards

    # Reshape to (num_rollouts * rollout_steps, num_envs)
    num_rollouts, rollout_steps, num_envs = dones.shape
    dones_flat = dones.reshape(-1, num_envs)
    rewards_flat = rewards.reshape(-1, num_envs)

    # Initialize episodic rewards tracker
    episodic_reward = np.zeros(num_envs)

    # Initialize lists to store steps and rewards
    steps = []
    episodic_rewards = []

    # Iterate over each step
    for idx, (done_step, reward_step) in enumerate(zip(dones_flat, rewards_flat)):
        # Accumulate rewards
        episodic_reward += np.array(reward_step)

        # Identify which environments are done
        done_envs = np.where(done_step)[0]
        if done_envs.size > 0:
            for env in done_envs:
                # Calculate the step index multiplied by number of environments
                step_index = idx * num_envs + env
                steps.append(step_index)
                # Store the episodic reward
                episodic_rewards.append(episodic_reward[env])
                # Reset the episodic reward for the environment
                episodic_reward[env] = 0.0

    return steps, episodic_rewards

def compute_moving_average(data, window_size):
    """Computes the moving average of the data with symmetric padding."""

    # Symmetrically pad the data
    pad_size = window_size // 2
    padded_data = np.pad(data, (pad_size, pad_size), mode='edge')

    # Convolve for mean
    window = np.ones(window_size) / window_size
    moving_avg_full = np.convolve(padded_data, window, mode='same')
    
    # Trim padded areas
    moving_avg = moving_avg_full[pad_size:-pad_size]
    return moving_avg

def compute_moving_std(data, window_size):
    """Computes the moving standard deviation of the data with symmetric padding."""

    # Get moving average
    moving_avg = compute_moving_average(data, window_size)

    # Compute std
    moving_avg_sq = compute_moving_average(data**2, window_size)
    moving_var = moving_avg_sq - moving_avg**2
    moving_std = np.sqrt(np.maximum(moving_var, 1e-10))
    return moving_std

def plot_moving_average(ax, steps, episodic_rewards, window_size, label, color):
    """Plots the moving average of episodic rewards with standard deviation bands."""

    # Sort the data by steps to ensure the order is correct
    sorted_indices = np.argsort(steps)
    sorted_steps = steps[sorted_indices]
    sorted_rewards = episodic_rewards[sorted_indices]

    # Compute moving average and std
    moving_avg = compute_moving_average(sorted_rewards, window_size)
    moving_std = compute_moving_std(sorted_rewards, window_size)
    upper_band = moving_avg + moving_std
    lower_band = moving_avg - moving_std

    # Plot moving average
    ax.plot(sorted_steps, moving_avg, label=label, color=color, lw=2)

    # Plot confidence bands
    ax.fill_between(sorted_steps, lower_band, upper_band, color=color, lw=0, alpha=0.15)

def plot_results(results_dict, window_size=100, title='Episodic Reward', colors=None):
    """Plots comparison results for multiple algorithms."""
    
    # Define color cycle
    if colors == 'gradient':
        indices = np.linspace(0, 255, len(results_dict.keys()), dtype=int)
        colors = np.array(plt.cm.viridis.colors)[indices]
    elif colors is None:
        colors = [
        '#636EFA',  # Blue
        '#EF533B',  # Red
        '#00CC96',  # Green
        '#AB63FA',  # Purple
        '#FFA15A',  # Orange
        '#19D3F3',  # Sky
        '#FF6692',  # Fuchsia
        '#B6E880',  # Lime
        '#FF97FF',  # Pink
        '#FECB52',  # Yellow
        ]

    color_cycle = {}
    for idx, alg in enumerate(results_dict.keys()):
        color_cycle[alg] = colors[idx % len(colors)]

    # Create Moving Average Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for alg_name, logs in results_dict.items():
        steps, episodic_rewards = process_logs(logs)
        steps = np.array(steps)
        episodic_rewards = np.array(episodic_rewards)
        plot_moving_average(ax, steps, episodic_rewards, window_size, label=alg_name, color=color_cycle[alg_name])

    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True)
    fig.tight_layout()
    plt.show()
