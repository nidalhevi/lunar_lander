import csv
import os
import matplotlib.pyplot as plt



def load_training_log(run_dir: str):
    """
    Loads one experiment's training log.

    Returns:
        generations: list of generation numbers
        overall_best: list of best-so-far fitness values
    """

    log_path = os.path.join(run_dir, "training_log.csv")

    generations = []
    overall_best = []

    with open(log_path, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            generations.append(int(row["generation"]))
            overall_best.append(float(row["overall_best"]))

    return generations, overall_best


def compare_experiments():
    """
    Reads all experiment folders inside runs/
    and plots their best-so-far fitness curves.
    """

    runs_dir = "runs"

    plt.figure()

    for run_name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, run_name)
        log_path = os.path.join(run_dir, "training_log.csv")

        if not os.path.isdir(run_dir):
            continue

        if not os.path.exists(log_path):
            continue

        generations, overall_best = load_training_log(run_dir)

        plt.plot(generations, overall_best, label=run_name)

    plt.xlabel("Generation")
    plt.ylabel("Best-so-far fitness")
    plt.title("Comparison of Mutation Strength Experiments")
    plt.legend()
    plt.grid(True)

    os.makedirs("runs/summary", exist_ok=True)
    plt.savefig("runs/summary/mutation_comparison.png")
    plt.show()


if __name__ == "__main__":
    compare_experiments()
    
    