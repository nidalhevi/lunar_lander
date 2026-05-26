# LunarLander Evolutionary Algorithm Solver

## Overview

This project implements an **Evolutionary Algorithm (EA)** to solve the **Gymnasium LunarLander-v3** environment.

Instead of using gradient-based reinforcement learning (e.g. backpropagation or Deep Q-Learning), this project evolves populations of candidate policies using:

* mutation,
* crossover,
* selection,
* elitism.

The project was developed for the course:

> *Topics in Computational Modelling: From Information Theory to Evolutionary Models*

---

# Project Goals

The main objective is to evolve an autonomous lunar landing controller capable of:

* stabilizing the spacecraft,
* controlling orientation,
* reducing velocity,
* landing safely.

The project also investigates:

* mutation strength,
* crossover usefulness,
* stochastic evaluation,
* evolutionary convergence,
* exploration vs exploitation.

---

# Evolutionary Computing Concepts Used

The implementation follows classical Evolutionary Algorithm principles:

| EA Concept         | Implementation                |
| ------------------ | ----------------------------- |
| Individual         | Neural-network policy         |
| Genome             | Vector of real-valued weights |
| Population         | List of genomes               |
| Fitness            | Average LunarLander reward    |
| Mutation           | Gaussian perturbation         |
| Recombination      | Uniform crossover             |
| Parent selection   | Tournament selection          |
| Survivor selection | Elitism                       |
| Evolutionary cycle | Generational replacement      |

The project is best described as a hybrid:

* **Evolution Strategy (ES)** +
* **Genetic Algorithm (GA)**

because it uses:

* real-valued genomes,
* Gaussian mutation,
* optional crossover.

---

# Project Structure

```text
lunar_lander/
│
├── src/
│   ├── env_eval.py
│   ├── policies.py
│   ├── evolution.py
│   ├── render_best.py
│   ├── plot_results.py
│   ├── run_experiments.py
│   ├── compare_experiments.py
│   ├── final_evaluation.py
│   └── crossover_experiments.py
│
├── runs/
│   ├── experiment folders
│   └── summary/
│
└── README.md
```

---

# File Descriptions

## `env_eval.py`

Handles interaction with Gymnasium:

* creates the LunarLander environment,
* runs episodes,
* computes rewards,
* evaluates genomes.

Important functions:

* `run_random_episode()`
* `run_policy_episode()`
* `evaluate_genome()`

---

## `policies.py`

Defines the neural-network controller and genome representation.

Main ideas:

* genomes are flat NumPy vectors,
* genomes are decoded into neural-network weights,
* the neural network maps observations to actions.

Neural network architecture:

```text
8 observations
→ 16 hidden neurons
→ 4 actions
```

Also contains:

* genome mutation utilities,
* genome decoding logic.

---

## `evolution.py`

Core evolutionary algorithm implementation.

Contains:

* population initialization,
* fitness evaluation,
* tournament selection,
* Gaussian mutation,
* uniform crossover,
* elitism,
* evolutionary training loop.

Main evolutionary cycle:

```text
Initialize population
→ Evaluate fitness
→ Select parents
→ Mutation / crossover
→ Create offspring
→ Survivor selection
→ Repeat
```

---

## `render_best.py`

Loads and visualizes the best evolved genome.

Useful for:

* qualitative analysis,
* observing landing behavior.

---

## `plot_results.py`

Plots training curves:

* generation best,
* average population fitness,
* overall best fitness.

---

## `run_experiments.py`

Runs mutation-strength experiments.

Example tested parameters:

```python
mutation_values = [0.01, 0.05, 0.10, 0.25]
```

---

## `compare_experiments.py`

Compares training curves across experiments.

Produces:

* multi-run comparison plots.

---

## `final_evaluation.py`

Evaluates saved genomes over many random seeds.

Outputs:

* mean reward,
* standard deviation,
* minimum reward,
* maximum reward.

This provides statistically meaningful evaluation.

---

## `crossover_experiments.py`

Compares:

* mutation-only EA,
* mutation + crossover EA.

---

# Neural Network Representation

Each genome encodes all neural-network parameters:

```text
w1: observation → hidden
b1: hidden biases
w2: hidden → output
b2: output biases
```

Genome size:

```text
212 parameters
```

---

# Selection Method

## Tournament Selection

Procedure:

1. Randomly select k individuals,
2. Choose the fittest.

Advantages:

* simple,
* efficient,
* adjustable selection pressure.

---

# Mutation Operator

## Gaussian Mutation

Mutation adds Gaussian noise:

```text
gene = gene + N(0, σ)
```

Where:

* `σ` = mutation standard deviation.

Mutation experiments were performed to analyze:

* exploration,
* convergence,
* stability.

---

# Crossover Operator

## Uniform Crossover

For each gene:

* inherit from parent 1 OR parent 2.

This allows:

* recombination of useful traits,
* increased diversity.

---

# Fitness Function

Fitness is computed as:

```text
Average total LunarLander reward over multiple episodes
```

Multiple seeds are used to reduce stochastic overfitting.

---

# Key Findings

## 1. Mutation Strength Matters

Small mutation:

* stable,
* slower exploration.

Large mutation:

* more exploration,
* less stability.

---

## 2. Overfitting to Seeds

Using fixed seeds during training caused policies to specialize to those environments.

Solution:

* randomize evaluation seeds.

---

## 3. Crossover Effects

Crossover sometimes improved:

* convergence speed,
* diversity.

However, mutation-only evolution also performed reasonably well.

---

# Installation

## Requirements

Python 3.10+

Install dependencies:

```bash
pip install "gymnasium[box2d]" numpy matplotlib
```

---

# Running the Project

## Train EA

```bash
python src/evolution.py
```

---

## Render Best Genome

```bash
python src/render_best.py
```

---

## Run Mutation Experiments

```bash
python src/run_experiments.py
```

---

## Compare Experiments

```bash
python src/compare_experiments.py
```

---

## Final Evaluation

```bash
python src/final_evaluation.py
```

---

# Future Improvements

Possible future extensions:

* adaptive mutation rates,
* self-adaptive evolution strategies,
* larger populations,
* deeper neural networks,
* linear policies,
* rule-based evolved policies,
* novelty search,
* multi-objective optimization,
* parallel evaluation.

---

# Theoretical Background

This project is grounded in:

* Evolutionary Algorithms,
* Evolution Strategies,
* Genetic Algorithms,
* Neuroevolution,
* Stochastic Optimization.

Core EA principles demonstrated:

* representation,
* selection,
* variation,
* population management,
* exploration/exploitation tradeoff,
* parameter tuning.

---

# Author

Nidal Ogur

Course project:

> Topics in Computational Modelling: From Information Theory to Evolutionary Models
