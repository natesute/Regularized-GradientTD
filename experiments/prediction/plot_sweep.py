import numpy as np
import pickle
from utils.weighting import WEIGHTINGS
from utils.config import PROBLEMS, STEPSIZES

# ---------------------
# Plotting Stepsize Sweep
# ---------------------

import matplotlib.pyplot as plt


# unpickle the collector object
with open('sweep.pkl', 'rb') as f:
    collector = pickle.load(f)

# Create subplots for each problem/representation
fig, axs = plt.subplots(len(PROBLEMS), 1, figsize=(10, 5 * len(PROBLEMS)))
if len(PROBLEMS) == 1:
    axs = [axs]

for i, problem in enumerate(PROBLEMS):
    env_name = problem['env'].__name__
    rep_name = problem['representation'].__name__
    ax = axs[i]
    ax.set_title(f'Stepsize Sweep in {env_name} with {rep_name}')
    ax.set_xlabel('Stepsize')
    ax.set_ylabel('RMSPBE')
    ax.set_xscale('log', base=2)  # Log scale for stepsize (base 2)

    for weighting_num, weighting in enumerate(WEIGHTINGS):
        mean_rmspbes = []
        stderr_rmspbes = []
        for stepsize in STEPSIZES:
            data_key = f"{env_name}-{rep_name}-{weighting.__name__}-{stepsize}"
            mean_curve, stderr_curve, _ = collector.getStats(data_key)
            mean_rmspbes.append(mean_curve.mean())
            stderr_rmspbes.append(stderr_curve.mean())
            # print(f"Stepsize: {stepsize}, Mean RMSPBE: {mean_rmspbes[-1]:.4f}, StdErr: {stderr_rmspbes[-1]:.4f}")

        # clip RMSPBE values for better visualization
        mean_rmspbes = np.clip(mean_rmspbes, None, 0.4)
        stderr_rmspbes = np.clip(stderr_rmspbes, None, 0.1)

        # plot the RMSPBE values for each stepsize
        ax.errorbar(STEPSIZES, mean_rmspbes, yerr=stderr_rmspbes, capsize=3, label=weighting.__name__)

    ax.legend()

plt.tight_layout()
plt.show()
fig.savefig(r'G:\My Drive\Regularized-GradientTD\figures\sweep.png')