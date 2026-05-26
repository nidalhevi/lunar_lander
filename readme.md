# LunarLander Evolutionary Algorithm Solver

## Overview

This project implements an **Evolutionary Algorithm (EA)** to solve the **Gymnasium LunarLander-v3** environment using **neuroevolution**.

Instead of training the controller with gradient-based reinforcement learning, the project evolves neural-network policies directly through:

* mutation,
* crossover,
* tournament selection,
* elitism,
* adaptive mutation scheduling.

The project also compares the EA solution with a **Deep Q-Network (DQN)** reinforcement learning baseline implemented in `rl_lander.ipynb`.

The project was developed for:

> *Topics in Computational Modelling: From Information Theory to Evolutionary Models*

---

# Main Features

## Evolutionary Algorithm Features

* Real-valued genome representation
* Neuroevolution
* Tournament selection
* Gaussian mutation
* Adaptive mutation scheduling
* Uniform crossover
* Elitism
* Multi-seed stochastic evaluation
* Experiment comparison framework
* Statistical evaluation system

## Reinforcement Learning Baseline

The repository also contains:

```text
rl_lander.ipynb
```

which implements a Deep Q-Network (DQN) LunarLander agent based on the Tutorial Horizon implementation.

This allows direct comparison between:

* Evolutionary Optimization,
* Reinforcement Learning.

---

# Environment

## LunarLander-v3

Observation space:

```text
8 continuous values
```

Actions:

```text
0 = do nothing
1 = left engine
2 = main engine
3 = right engine
```

The goal is to maximize cumulative landing reward.

---

# Evolutionary Algorithm

## Neural Network Representation

The evolved policy is a neural network:

```text
8 → 32 → 4
```

where:

* 8 inputs = LunarLander observations,
* 32 hidden neurons,
* 4 output action scores.

The hidden layer uses:

```python
tanh()
```

The selected action is:

```python
argmax(action_scores)
```

---

# Genome Representation

Each individual is a real-valued vector containing:

* input-to-hidden weights,
* hidden biases,
* hidden-to-output weights,
* output biases.

Genome size:

```text
420 parameters
```

---

# Evolutionary Operators

## Selection

Tournament selection:

```text
k = 3
```

---

## Mutation

Gaussian mutation with mutation probability:

```python
gene += N(0, σ)
```

Mutation rate:

```text
0.10
```

---

## Adaptive Mutation

Mutation standard deviation decreases linearly:

```text
σ_initial = 0.30
σ_final   = 0.03
```

This creates:

* strong early exploration,
* late-stage fine-tuning.

---

## Crossover

Uniform crossover:

* each gene independently inherited from one parent.

---

## Elitism

Best individuals survive directly into the next generation.

---

# Final EA Configuration

Main successful setup:

| Parameter               | Value    |
| ----------------------- | -------- |
| Population size         | 100      |
| Generations             | 150      |
| Hidden neurons          | 32       |
| Tournament size         | 3        |
| Elite count             | 2        |
| Mutation rate           | 0.10     |
| Initial mutation std    | 0.30     |
| Final mutation std      | 0.03     |
| Episodes per evaluation | 10       |
| Adaptive mutation       | enabled  |
| Crossover               | optional |

---

# Main Results

## Mutation-only EA

Final results:

```text
Best-so-far fitness: 298.17
Final average population fitness: 274.66
```

---

## EA with Crossover

Final results:

```text
Best-so-far fitness: 294.31
Final average population fitness: 244.20
```

---

## RL DQN Baseline

The DQN model solved the environment in:

```text
615 episodes
Average score ≈ 200
```

---

# Key Findings

## 1. Adaptive Mutation Was Critical

Adaptive mutation scheduling dramatically improved:

* convergence,
* exploration,
* late-stage stability.

---

## 2. Small Mutation Values Failed

Mutation values:

* `0.01`,
* `0.05`

often stagnated early.

Larger adaptive mutation schedules performed much better.

---

## 3. Multi-Seed Evaluation Was Necessary

Early versions overfit to fixed seeds.

Using random evaluation seeds greatly improved:

* robustness,
* generalization.

---

## 4. Crossover Was Useful but Not Dominant

Both:

* mutation-only EA,
* mutation + crossover EA

successfully solved LunarLander.

Mutation-only evolution slightly outperformed crossover in the final experiments, though crossover still produced high-performing policies.

---

# Reinforcement Learning Comparison

## Evolutionary Algorithm

Policy improvement through:

* mutation,
* selection,
* recombination.

No gradients used.

---

## DQN

Policy improvement through:

* temporal-difference learning,
* replay memory,
* gradient descent,
* Bellman updates.

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
│   ├── crossover_experiments.py
│   ├── plot_dqn_results.py
│   └── plot_final_comparison.py
│
├── runs/
│   ├── mutation_only/
│   ├── mutation_plus_crossover/
│   ├── rl/
│   └── summary/
│
├── rl_lander.ipynb
│
└── README.md
```

---

# Important Files

## `evolution.py`

Main EA implementation:

* population initialization,
* selection,
* mutation,
* crossover,
* adaptive mutation scheduling,
* elitism,
* training loop.

---

## `policies.py`

Defines:

* neural-network architecture,
* genome decoding,
* policy inference.

---

## `env_eval.py`

Runs LunarLander episodes and evaluates genomes.

---

## `crossover_experiments.py`

Compares:

* mutation-only EA,
* crossover EA.

---

## `rl_lander.ipynb`

DQN reinforcement learning baseline.

---

# Generated Figures

The project generates:

* EA training curves,
* mutation comparison plots,
* DQN learning curves,
* EA vs DQN comparison plots.

Saved in:

```text
runs/summary/
```

---

# Installation

## Requirements

```bash
pip install gymnasium[box2d] numpy matplotlib torch
```

---

# Running the Evolutionary Algorithm

## Train EA

```bash
python src/evolution.py
```

---

## Run Mutation Experiments

```bash
python src/run_experiments.py
```

---

## Run Crossover Experiments

```bash
python src/crossover_experiments.py
```

---

## Render Best Policy

```bash
python src/render_best.py
```

---

## Compare Experiments

```bash
python src/compare_experiments.py
```

---

## Final Statistical Evaluation

```bash
python src/final_evaluation.py
```

---

# Running the DQN Baseline

Open:

```text
rl_lander.ipynb
```

and execute all notebook cells.

---

# Main Contributions

This project demonstrates:

* neuroevolution,
* real-valued evolutionary optimization,
* adaptive mutation scheduling,
* stochastic policy evaluation,
* EA vs RL comparison,
* direct policy evolution without backpropagation.

---

# Future Improvements

Possible extensions:

* deeper neural networks,
* self-adaptive mutation rates,
* novelty search,
* parallel evaluation,
* multi-objective optimization,
* recurrent neural policies,
* rule-based evolved policies,
* hybrid EA + RL methods.

---

# Author

Nidal Ogur

Course Project:

> Topics in Computational Modelling: From Information Theory to Evolutionary Models
