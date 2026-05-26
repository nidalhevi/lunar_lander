from evolution import train


mutation_values = [0.01, 0.05, 0.10, 0.25]

for mutation_std in mutation_values:
    print()
    print("=" * 50)
    print(f"Running experiment with mutation_std = {mutation_std}")
    print("=" * 50)

    train(
        population_size=100,
        generations=150,
        mutation_std=mutation_std,
        run_name=f"mutation_{mutation_std}",
        use_adaptive_mutation=False,
    )