import numpy as np
from policies import random_genome
from env_eval import run_policy_episode
import csv
import os


def initialize_population(population_size: int) -> list[np.ndarray]:
    """Create the first generation of random genomes."""
    return [random_genome() for _ in range(population_size)]


def evaluate_genome(genome: np.ndarray, n_episodes: int = 10) -> float:
    rewards = []

    seeds = np.random.randint(0, 100000, size=n_episodes)

    for seed in seeds:
        reward = run_policy_episode(genome, render=False, seed=int(seed))
        rewards.append(reward)

    return float(np.mean(rewards))


def evaluate_population(population: list[np.ndarray]) -> list[float]:
    """Evaluate every genome in the population."""
    return [evaluate_genome(genome) for genome in population]


def tournament_selection(
    population: list[np.ndarray],
    fitnesses: list[float],
    tournament_size: int = 3,
) -> np.ndarray:
    """Select one parent using tournament selection."""
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)

    best_index = selected_indices[0]

    for index in selected_indices:
        if fitnesses[index] > fitnesses[best_index]:
            best_index = index

    return population[best_index].copy()


def mutate(genome: np.ndarray, mutation_std: float = 0.1, mutation_rate: float = 0.1) -> np.ndarray:
    """Apply Gaussian mutation to a genome."""
    child = genome.copy()
    mask = np.random.random(size=child.shape) < mutation_rate
    noise = np.random.normal(0, mutation_std, size=child.shape)
    child[mask] += noise[mask]
    return child

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """
    Creates one child by combining genes from two parents.

    This is uniform crossover:
    for each gene, the child randomly inherits either
    the value from parent1 or the value from parent2.
    """

    mask = np.random.rand(len(parent1)) < 0.5

    child = np.where(mask, parent1, parent2)

    return child

def make_next_generation(
    population: list[np.ndarray],
    fitnesses: list[float],
    elite_count: int = 2,
    mutation_std: float = 0.1,
    use_crossover: bool = False,
) -> list[np.ndarray]:
    """Create the next generation using elitism, selection, and mutation."""
    next_population = []

    sorted_indices = np.argsort(fitnesses)[::-1]

    for i in range(elite_count):
        elite_index = sorted_indices[i]
        next_population.append(population[elite_index].copy())

    while len(next_population) < len(population):
        if use_crossover:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2)
        else:
            parent = tournament_selection(population, fitnesses)
            child = parent.copy()

        child = mutate(child, mutation_std)
        next_population.append(child)

    return next_population

def get_adaptive_mutation_std(
    initial_std: float,
    final_std: float,
    generation: int,
    generations: int,
) -> float:
    """
    Linearly decrease mutation strength during evolution.

    Early in training, mutation is large to explore.
    Later in training, mutation is smaller to fine-tune.
    """

    progress = generation / (generations - 1)

    current_std = initial_std + progress * (final_std - initial_std)

    return current_std

def train(
    population_size: int = 100,
    generations: int = 150,
    mutation_std: float = 0.30,
    final_mutation_std: float = 0.03,
    run_name: str = "default",
    use_crossover: bool = False,
    use_adaptive_mutation: bool = True,
):
    """Main evolutionary training loop."""
    population = initialize_population(population_size)
    
    history = []

    best_genome = None
    best_fitness = -float("inf")

    for generation in range(generations):
        fitnesses = evaluate_population(population)

        generation_best = max(fitnesses)
        generation_average = float(np.mean(fitnesses))

        best_index = int(np.argmax(fitnesses))

        if generation_best > best_fitness:
            best_fitness = generation_best
            best_genome = population[best_index].copy()


        history.append({
            "generation": generation + 1,
            "best": generation_best,
            "average": generation_average,
            "overall_best": best_fitness,
        })


        

        if use_adaptive_mutation:
            current_mutation_std = get_adaptive_mutation_std(
                initial_std=mutation_std,
                final_std=final_mutation_std,
                generation=generation,
                generations=generations,
            )
        else:
            current_mutation_std = mutation_std

        population = make_next_generation(
            population,
            fitnesses,
            elite_count=2,
            mutation_std=current_mutation_std,
            use_crossover=use_crossover,
        )
        
        print(
            f"Generation {generation + 1} | "
            f"Best: {generation_best:.2f} | "
            f"Average: {generation_average:.2f} | "
            f"Overall best: {best_fitness:.2f} | "
            f"Mutation std: {current_mutation_std:.4f}"
        )

    
    
    run_dir = f"runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    
    with open(f"{run_dir}/training_log.csv", "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["generation", "best", "average", "overall_best"]
        )

        writer.writeheader()
        writer.writerows(history)

    np.save(f"{run_dir}/best_genome.npy", best_genome)

    return best_genome, best_fitness


if __name__ == "__main__":
    train()