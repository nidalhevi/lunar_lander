"""
env_eval.py

This file contains functions for running LunarLander episodes.

At this stage, the goal is not yet to evolve anything. The goal is to:
    1. Create the Gymnasium LunarLander environment.
    2. Run one episode with random actions.
    3. Run one episode with a genome-based neural policy.
    4. Return the total reward as the fitness value.

Later, the evolutionary algorithm will call these functions many times to
measure how good each genome is.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from policies import policy_action, random_genome


ENV_NAME = "LunarLander-v3"


def make_env(render: bool = False) -> gym.Env:
    """
    Create a LunarLander environment.

    Args:
        render: If True, open a visual window and show the episode.
                If False, run without graphics, which is faster.

    Returns:
        A Gymnasium environment object.
    """

    # Gymnasium uses render_mode="human" to display the environment.
    render_mode = "human" if render else None

    # Make and return the environment.
    return gym.make(ENV_NAME, render_mode=render_mode)


def run_random_episode(render: bool = False, seed: int | None = None) -> float:
    """
    Run one full episode using completely random actions.

    This is useful as a baseline. A random policy should usually perform badly.

    Args:
        render: If True, visually display the episode.
        seed: Optional random seed for reproducibility.

    Returns:
        Total reward collected during the episode.
    """

    # Create the environment.
    env = make_env(render=render)

    # Reset starts a new episode.
    # Gymnasium returns observation and info.
    observation, info = env.reset(seed=seed)

    # Track the reward accumulated during the episode.
    total_reward = 0.0

    # Gymnasium episodes end in two possible ways:
    # terminated = natural terminal state, e.g. crash or successful landing.
    # truncated = time limit reached.
    done = False

    while not done:
        # Randomly sample one valid action from the action space.
        action = env.action_space.sample()

        # Apply the action to the environment.
        observation, reward, terminated, truncated, info = env.step(action)

        # Add the reward from this step to the episode total.
        total_reward += float(reward)

        # The episode is done if either termination condition is true.
        done = terminated or truncated

    # Always close the environment when finished.
    env.close()

    return total_reward


def run_policy_episode(
    genome: np.ndarray,
    render: bool = False,
    seed: int | None = None,
) -> float:
    """
    Run one full episode using a genome-controlled neural-network policy.

    This function is the future fitness evaluation for the evolutionary algorithm.

    Args:
        genome: Real-valued vector encoding the neural-network policy.
        render: If True, visually display the episode.
        seed: Optional random seed for reproducibility.

    Returns:
        Total reward collected during the episode.
    """

    # Create the environment.
    env = make_env(render=render)

    # Reset the environment to start a new episode.
    observation, info = env.reset(seed=seed)

    total_reward = 0.0
    done = False

    while not done:
        # Use the genome-based neural network to choose the action.
        action = policy_action(observation=np.asarray(observation), genome=genome)

        # Step the environment forward using that action.
        observation, reward, terminated, truncated, info = env.step(action)

        # Accumulate reward.
        total_reward += float(reward)

        # Stop when the episode terminates or reaches the time limit.
        done = terminated or truncated

    env.close()

    return total_reward


def evaluate_genome(
    genome: np.ndarray,
    n_episodes: int = 3,
    seed: int = 0,
) -> float:
    """
    Evaluate one genome over several episodes and return average reward.

    LunarLander is stochastic, so one episode can be misleading. Averaging over
    multiple episodes gives a more reliable fitness value.

    Args:
        genome: Genome to evaluate.
        n_episodes: Number of episodes used for evaluation.
        seed: Base seed. Episode i uses seed + i.

    Returns:
        Average total reward across episodes.
    """

    rewards = []

    for i in range(n_episodes):
        # Use a different seed for each episode.
        episode_seed = seed + i

        # Run one policy episode and store the reward.
        reward = run_policy_episode(genome=genome, render=False, seed=episode_seed)
        rewards.append(reward)

    # Convert to float so the result is a plain Python number.
    return float(np.mean(rewards))


def demo_random_policy() -> None:
    """
    Run several random episodes and print their rewards.

    This confirms that the environment works.
    """

    print("Running random policy baseline...")

    rewards = []

    for episode in range(5):
        reward = run_random_episode(render=False, seed=episode)
        rewards.append(reward)
        print(f"Random episode {episode + 1}: reward = {reward:.2f}")

    print(f"Average random reward: {np.mean(rewards):.2f}")


def demo_genome_policy() -> None:
    """
    Create one random genome and evaluate it.

    This confirms that policies.py and env_eval.py work together.
    """

    print("\nRunning one random neural-network genome...")

    rng = np.random.default_rng(seed=42)
    genome = random_genome(rng=rng)

    fitness = evaluate_genome(genome=genome, n_episodes=3, seed=100)

    print(f"Genome length: {len(genome)}")
    print(f"Average reward over 3 episodes: {fitness:.2f}")

    # To visually watch the random neural policy, uncomment this line:
    # run_policy_episode(genome=genome, render=True, seed=123)


if __name__ == "__main__":
    # First test: random actions.
    demo_random_policy()

    # Second test: random neural-network genome.
    demo_genome_policy()
