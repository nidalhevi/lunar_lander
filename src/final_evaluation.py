import csv
import os
import numpy as np

from env_eval import run_policy_episode


def evaluate_saved_genome(genome_path: str, n_episodes: int = 30) -> dict:
    """
    Evaluates one saved genome over many episodes.
    """

    genome = np.load(genome_path)

    rewards = []

    for seed in range(n_episodes):
        reward = run_policy_episode(genome, render=False, seed=1000 + seed)
        rewards.append(reward)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
    }


def evaluate_all_runs():
    """
    Evaluates the best genome from every experiment folder.
    """

    runs_dir = "runs"
    results = []

    for run_name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, run_name)
        genome_path = os.path.join(run_dir, "best_genome.npy")

        if not os.path.isdir(run_dir):
            continue

        if not os.path.exists(genome_path):
            continue

        stats = evaluate_saved_genome(genome_path, n_episodes=30)

        row = {
            "run_name": run_name,
            **stats,
        }

        results.append(row)

        print(
            f"{run_name}: "
            f"mean={stats['mean_reward']:.2f}, "
            f"std={stats['std_reward']:.2f}, "
            f"min={stats['min_reward']:.2f}, "
            f"max={stats['max_reward']:.2f}"
        )

    os.makedirs("runs/summary", exist_ok=True)

    with open("runs/summary/final_evaluation.csv", "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "run_name",
                "mean_reward",
                "std_reward",
                "min_reward",
                "max_reward",
            ],
        )

        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    evaluate_all_runs()