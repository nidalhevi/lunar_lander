import numpy as np
from env_eval import run_policy_episode


RUN_NAME = "default"
GENOME_PATH = f"runs/{RUN_NAME}/best_genome.npy"


if __name__ == "__main__":
    genome = np.load(GENOME_PATH)

    rewards = []

    for seed in range(10):
        reward = run_policy_episode(genome, render=False, seed=seed)
        rewards.append(reward)
        print(f"Seed {seed}: reward = {reward:.2f}")

    print()
    print(f"Average reward over 10 seeds: {np.mean(rewards):.2f}")
    print(f"Best seed reward: {np.max(rewards):.2f}")
    print(f"Worst seed reward: {np.min(rewards):.2f}")

    best_seed = int(np.argmax(rewards))
    print(f"Rendering best seed: {best_seed}")

    run_policy_episode(genome, render=True, seed=best_seed)