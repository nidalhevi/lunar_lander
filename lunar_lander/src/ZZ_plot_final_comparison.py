import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_ea_score(run_name: str) -> float:
    with open("runs/summary/final_evaluation.csv", "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["run_name"] == run_name:
                return float(row["mean_reward"])

    raise ValueError(f"Run not found: {run_name}")


def get_dqn_final_score() -> float:
    last_average = None

    with open("runs/rl/dqn_scores.csv", "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            last_average = float(row["average_100"])

    if last_average is None:
        raise ValueError("No DQN scores found.")

    return last_average


ea_mutation_only = get_ea_score("mutation_only")
ea_crossover = get_ea_score("mutation_plus_crossover")
dqn_score = get_dqn_final_score()

methods = [
    "EA mutation-only",
    "EA crossover",
    "DQN",
]

scores = [
    ea_mutation_only,
    ea_crossover,
    dqn_score,
]

plt.figure()
plt.bar(methods, scores)

plt.ylabel("Final average reward")
plt.title("Final Performance Comparison: EA vs DQN")
plt.grid(axis="y")

os.makedirs("runs/summary", exist_ok=True)
plt.savefig("runs/summary/ea_vs_dqn_comparison.png")
plt.close()