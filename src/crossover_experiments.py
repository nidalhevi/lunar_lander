from evolution import train


train(
    population_size=100,
    generations=150,
    mutation_std=0.3,
    final_mutation_std=0.03,
    run_name="mutation_only",
    use_crossover=False,
    use_adaptive_mutation=True,
)

train(
    population_size=100,
    generations=150,
    mutation_std=0.3,
    final_mutation_std=0.03,
    run_name="mutation_plus_crossover",
    use_crossover=True,
    use_adaptive_mutation=True,
)