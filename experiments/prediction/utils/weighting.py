import numpy as np

def prob_features(state_space):
    num_states, num_features = state_space.shape
    probability_state_space = np.zeros_like(state_space)

    for feature_idx in range(num_features):
        # Extract unique values and their counts for the current feature
        unique_values, counts = np.unique(state_space[:, feature_idx], return_counts=True)
        probabilities = counts / num_states  # Calculate probabilities

        # Map probabilities back to the state space
        for i, value in enumerate(unique_values):
            probability_state_space[:, feature_idx][state_space[:, feature_idx] == value] = probabilities[i]

    return probability_state_space

def weighted_features(state_space):
    probability_state_space = prob_features(state_space)
    weighted_state_space = probability_state_space * state_space
    weighted_state_space = np.log(1 + weighted_state_space)
    return weighted_state_space

def log_features(state_space):
    prob_state_space = prob_features(state_space)
    log_state_space = np.log(1 + prob_state_space)
    return log_state_space

def regular(state_space):
    return state_space

WEIGHTINGS = [regular, log_features, weighted_features, prob_features]