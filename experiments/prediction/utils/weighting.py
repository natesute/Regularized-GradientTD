import numpy as np

def prob_features(state_space):
    num_states, num_features = state_space.shape
    probability_state_space = np.zeros_like(state_space)

    for feature_idx in range(num_features):
        # extract unique values and their counts for the current feature
        unique_values, counts = np.unique(state_space[:, feature_idx], return_counts=True)
        probabilities = counts / num_states  # Calculate probabilities

        # map probabilities back to the state space
        for i, value in enumerate(unique_values):
            probability_state_space[:, feature_idx][state_space[:, feature_idx] == value] = probabilities[i]

    return probability_state_space

def weighted_features(state_space):
    probability_state_space = prob_features(state_space)
    weighted_state_space = probability_state_space * state_space
    return weighted_state_space

def log_features(state_space):
    prob_state_space = prob_features(state_space)
    prob_state_space = np.where(prob_state_space == 0, np.exp(-5), prob_state_space)  # replace 0 with exp(-5)
    log_state_space = np.log(prob_state_space)
    return log_state_space

def regular(state_space):
    return state_space

WEIGHTINGS = [regular, log_features]