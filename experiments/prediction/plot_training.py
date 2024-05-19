import numpy as np
import pickle
from utils.weighting import WEIGHTINGS
from utils.config import PROBLEMS
import matplotlib.pyplot as plt
from agents.GTD2 import GTD2

# ---------------------
# Plotting the bar plot
# ---------------------

# Load the collector object
with open('training.pkl', 'rb') as f:
    collector = pickle.load(f)

LEARNERS = [GTD2]

COLORS = {
    0: 'b',
    1: 'r',
    2: 'g',
    3: 'c',
    4: 'm',
    5: 'y',
}

# Setting up the figure and axes
fig, axs = plt.subplots(len(PROBLEMS), 1, figsize=(10, 5 * len(PROBLEMS)))

# Ensure `axs` is always an array, even with one element
if len(PROBLEMS) == 1:
    axs = [axs]

for i, problem in enumerate(PROBLEMS):
    for Learner in LEARNERS:
        learner_name = Learner.__name__
        env_name = problem['env'].__name__
        rep_name = problem['representation'].__name__
        print("env name:", env_name)
        print("rep_name:", rep_name)

        # This axis
        ax = axs[i]
        ax.set_title(f'Learning Curve in {env_name} with {rep_name}')
        ax.set_xlabel('Steps')
        ax.set_ylabel('RMSPBE')
        
        for weighting_num, weighting in enumerate(WEIGHTINGS):
            data_key = f'{env_name}-{rep_name}-{weighting.__name__}-{learner_name}'
            mean_curve, stderr_curve, _ = collector.getStats(data_key)

            # Plotting the mean RMSPBE over steps with error bars
            steps = range(len(mean_curve))  # Assuming mean_curve length matches the number of steps
            ax.errorbar(steps, mean_curve, yerr=stderr_curve, label=f'{learner_name} weighting {weighting.__name__}', color=COLORS[weighting_num])

        # Adding a legend to each subplot
        ax.legend()

# Adjusting layout to prevent overlap
plt.tight_layout()
plt.show()
fig.savefig(r'G:\My Drive\Regularized-GradientTD\figures\training.png')