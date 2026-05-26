"""
policies.py

This file contains the policy/controller code for the LunarLander project.

In this project, an evolutionary algorithm will optimize a population of
"genomes". Each genome is just a long NumPy array of real numbers.

Those numbers are decoded into the weights and biases of a small neural
network. The neural network receives the LunarLander observation and chooses
one of the four possible actions.

Important idea:
    The neural network is NOT trained with backpropagation.
    Its weights are evolved by mutation, selection, and later crossover.
"""

from __future__ import annotations

import numpy as np


# LunarLander-v3 gives the agent 8 observation values:
# 0: x position
# 1: y position
# 2: x velocity
# 3: y velocity
# 4: lander angle
# 5: angular velocity
# 6: left leg contact, 0 or 1
# 7: right leg contact, 0 or 1
OBS_DIM = 8

# Number of neurons in the hidden layer.
# This is small on purpose so the evolutionary search is not too expensive.
HIDDEN_DIM = 32

# LunarLander-v3 has 4 discrete actions:
# 0: do nothing
# 1: fire left orientation engine
# 2: fire main engine
# 3: fire right orientation engine
ACT_DIM = 4


def genome_size() -> int:
    """
    Return the number of real-valued genes needed for one neural policy.

    The network has this structure:
        observation -> hidden layer -> action scores

    Parameters stored in the genome:
        w1: weights from observation layer to hidden layer
        b1: bias values for hidden layer
        w2: weights from hidden layer to output/action layer
        b2: bias values for output/action layer

    Returns:
        Total number of values in the flat genome.
    """

    # Number of values in the first weight matrix.
    input_to_hidden_weights = OBS_DIM * HIDDEN_DIM

    # Number of hidden-layer bias values.
    hidden_biases = HIDDEN_DIM

    # Number of values in the second weight matrix.
    hidden_to_output_weights = HIDDEN_DIM * ACT_DIM

    # Number of output-layer bias values.
    output_biases = ACT_DIM

    return (
        input_to_hidden_weights
        + hidden_biases
        + hidden_to_output_weights
        + output_biases
    )


def random_genome(rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Create one random genome.

    This is used to initialize the first evolutionary population.

    Args:
        rng: Optional NumPy random generator. Passing this makes experiments
             reproducible because the same seed gives the same genomes.

    Returns:
        A NumPy array containing all neural-network parameters as a flat vector.
    """

    # Use a fresh random generator if the caller did not provide one.
    if rng is None:
        rng = np.random.default_rng()

    # Small initial values are usually better than very large values because
    # very large neural-network weights can make actions unstable at the start.
    return rng.normal(loc=0.0, scale=0.5, size=genome_size())


def genome_to_weights(genome: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode a flat genome into neural-network weights and biases.

    Args:
        genome: Flat vector of real numbers.

    Returns:
        w1, b1, w2, b2:
            w1 has shape (8, 16)
            b1 has shape (16,)
            w2 has shape (16, 4)
            b2 has shape (4,)
    """

    # Check that the genome has exactly the expected length.
    expected_size = genome_size()
    if genome.shape[0] != expected_size:
        raise ValueError(
            f"Genome has length {genome.shape[0]}, but expected {expected_size}."
        )

    # idx tracks where we are while slicing the flat genome.
    idx = 0

    # First layer weights: observation -> hidden.
    w1_end = idx + OBS_DIM * HIDDEN_DIM
    w1 = genome[idx:w1_end].reshape(OBS_DIM, HIDDEN_DIM)
    idx = w1_end

    # Hidden layer biases.
    b1_end = idx + HIDDEN_DIM
    b1 = genome[idx:b1_end]
    idx = b1_end

    # Second layer weights: hidden -> action scores.
    w2_end = idx + HIDDEN_DIM * ACT_DIM
    w2 = genome[idx:w2_end].reshape(HIDDEN_DIM, ACT_DIM)
    idx = w2_end

    # Output layer biases.
    b2_end = idx + ACT_DIM
    b2 = genome[idx:b2_end]

    return w1, b1, w2, b2


def policy_action(observation: np.ndarray, genome: np.ndarray) -> int:
    """
    Choose an action for the current LunarLander observation.

    This function converts the genome into a neural network, runs a forward
    pass, and returns the action with the highest score.

    Args:
        observation: Current environment observation, shape (8,).
        genome: Flat vector containing neural-network parameters.

    Returns:
        Integer action in {0, 1, 2, 3}.
    """

    # Decode the genome into matrices and vectors.
    w1, b1, w2, b2 = genome_to_weights(genome)

    # First neural-network layer.
    # The tanh activation keeps hidden values between -1 and 1.
    hidden = np.tanh(observation @ w1 + b1)

    # Output layer. These are not probabilities; they are action scores.
    action_scores = hidden @ w2 + b2

    # Pick the action with the largest score.
    return int(np.argmax(action_scores))


def mutate_genome(
    genome: np.ndarray,
    mutation_std: float = 0.1,
    mutation_rate: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Create a mutated copy of a genome using Gaussian mutation.

    This is an evolutionary operator, not neural-network training.

    Args:
        genome: Parent genome.
        mutation_std: Standard deviation of the Gaussian noise.
        mutation_rate: Probability that each gene is mutated.
        rng: Optional random generator for reproducibility.

    Returns:
        A new mutated genome. The original genome is not changed.
    """

    if rng is None:
        rng = np.random.default_rng()

    # Copy the genome so we do not accidentally change the parent.
    child = genome.copy()

    # Boolean mask: True means this gene will be mutated.
    mutation_mask = rng.random(size=child.shape) < mutation_rate

    # Gaussian noise to add to selected genes.
    noise = rng.normal(loc=0.0, scale=mutation_std, size=child.shape)

    # Apply mutation only where the mask is True.
    child[mutation_mask] += noise[mutation_mask]

    return child
