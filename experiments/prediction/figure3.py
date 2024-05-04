import numpy as np

# source found at: https://github.com/andnp/RlGlue
from RlGlue import RlGlue
from utils.Collector import Collector
from utils.policies import actionArrayToPolicy, matrixToPolicy
from utils.rl_glue import RlGlueCompatWrapper
from utils.errors import buildRMSPBE

from environments.RandomWalk import RandomWalk, TabularRep, DependentRep, InvertedRep
from environments.Boyan import Boyan, BoyanRep
from environments.Baird import Baird, BairdRep

from agents.TD import TD
from agents.TDC import TDC
from agents.HTD import HTD
from agents.GTD2 import GTD2
from agents.TDRC import TDRC
from agents.Vtrace import Vtrace

RUNS = 50

PROBLEMS = [
    # 5-state random walk environment with tabular features
    {
        'env': RandomWalk,
        'representation': TabularRep,
        # go LEFT 40% of the time
        'target': actionArrayToPolicy([0.4, 0.6]),
        # take each action equally
        'behavior': actionArrayToPolicy([0.5, 0.5]),
        'gamma': 1.0,
        'steps': 3000,
        # hardcode stepsizes found from parameter study
        'stepsizes': {
            'TD': 0.03125,
            'TDRC': 0.03125,
            'TDC': 0.0625,
            'GTD2': 0.03125,
            'GTDW': 0.015625,
            'HTD': 0.03125,
            'Vtrace': 0.03125,
        }
    },
    # 5-state random walk environment with dependent features
    {
        'env': RandomWalk,
        'representation': DependentRep,
        # go LEFT 40% of the time
        'target': actionArrayToPolicy([0.4, 0.6]),
        # take each action equally
        'behavior': actionArrayToPolicy([0.5, 0.5]),
        'gamma': 1.0,
        'steps': 3000,
        # hardcode stepsizes found from parameter study
        'stepsizes': {
            'TD': 0.03125,
            'TDRC': 0.03125,
            'TDC': 0.0625,
            'GTD2': 0.0625,
            'GTDW': 0.03125,
            'HTD': 0.03125,
            'Vtrace': 0.03125,
        }
    },
    # 5-state random walk environment with inverted features
    {
        'env': RandomWalk,
        'representation': InvertedRep,
        # go LEFT 40% of the time
        'target': actionArrayToPolicy([0.4, 0.6]),
        # take each action equally
        'behavior': actionArrayToPolicy([0.5, 0.5]),
        'gamma': 1.0,
        'steps': 3000,
        # hardcode stepsizes found from parameter study
        'stepsizes': {
            'TD': 0.125,
            'TDRC': 0.125,
            'TDC': 0.125,
            'GTD2': 0.125,
            'GTDW': 0.015625,
            'HTD': 0.125,
            'Vtrace': 0.125,
        }
    },
    # Boyan's chain
    {
        'env': Boyan,
        'representation': BoyanRep,
        # go LEFT 40% of the time
        'target': matrixToPolicy([[.5, .5]] * 10 + [[1., 0.]] * 2),
        # take each action equally
        'behavior': matrixToPolicy([[.5, .5]] * 10 + [[1., 0.]] * 2),
        'gamma': 1.0,
        'steps': 10000,
        # hardcode stepsizes found from parameter study
        'stepsizes': {
            'TD': 0.0625,
            'TDRC': 0.0625,
            'TDC': 0.5,
            'GTD2': 0.5,
            'GTDW': 0.0078125,
            'HTD': 0.0625,
            'Vtrace': 0.0625,
        }
    },
    # Baird's Counter-example domain
    {
        'env': Baird,
        'representation': BairdRep,
        # go LEFT 40% of the time
        'target': actionArrayToPolicy([0., 1.]),
        # take each action equally
        'behavior': actionArrayToPolicy([6/7, 1/7]),
        'starting_condition': np.array([1, 1, 1, 1, 1, 1, 1, 10]),
        'gamma': 0.99,
        'steps': 20000,
        # hardcode stepsizes found from parameter study
        'stepsizes': {
            'TD': 0.00390625,
            'TDRC': 0.015625,
            'TDC': 0.00390625,
            'GTD2': 0.00390625,
            'GTDW': 0.00390625,
            'HTD': 0.00390625,
            'Vtrace': 0.00390625,
        }
    },
]

STEPSIZES = [2**(i) for i in range(-9, -3)]  # [2^-6, 2^-5, ..., 2^5, 2^6]
LEARNER = GTD2

# -----------------------------------
# Collect Data for Stepsize Sweep
# -----------------------------------

collector = Collector()

for run in range(RUNS):
    for problem in PROBLEMS:
        env_name = problem['env'].__name__
        rep_name = problem['representation'].__name__
        
        for stepsize in STEPSIZES:
            # Reproducibility and reset
            np.random.seed(run)

            # Environment and representation
            Env = problem['env']
            env = Env()
            target = problem['target']
            Rep = problem['representation']
            rep = Rep(weighted=True)

            print(f"Run {run}, Env: {env_name}, Rep: {rep_name}, Stepsize: {stepsize}")

            X, P, R, D = env.getXPRD(target, rep)
            RMSPBE = buildRMSPBE(X, P, R, D, problem['gamma'])

            # Learner with current stepsize
            learner = LEARNER(rep.features(), {
                'gamma': problem['gamma'],
                'alpha': stepsize,
                'beta': 1,
            })

            # Agent and experiment setup (similar to previous experiments)
            agent = RlGlueCompatWrapper(learner, problem['behavior'], problem['target'], rep.encode)
            glue = RlGlue(agent, env)
            glue.start()

            for step in range(problem['steps']):
                _, _, _, terminal = glue.step()
                if terminal:
                    glue.start()

                if step % 100 == 0:
                    w = learner.getWeights()
                    rmspbe = RMSPBE(w)
                    collector.collect(f"{env_name}-{rep_name}-{stepsize}", rmspbe)

            collector.reset()

# ---------------------
# Plotting Stepsize Sweep
# ---------------------

import matplotlib.pyplot as plt

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

    mean_rmspbes = []
    stderr_rmspbes = []
    for stepsize in STEPSIZES:
        data_key = f"{env_name}-{rep_name}-{stepsize}"
        mean_curve, stderr_curve, _ = collector.getStats(data_key)
        mean_rmspbes.append(mean_curve.mean())
        stderr_rmspbes.append(stderr_curve.mean())
        print(f"Stepsize: {stepsize}, Mean RMSPBE: {mean_rmspbes[-1]:.4f}, StdErr: {stderr_rmspbes[-1]:.4f}")
    ax.errorbar(STEPSIZES, mean_rmspbes, yerr=stderr_rmspbes, capsize=3)

plt.tight_layout()
plt.show()
fig.savefig(r'G:\My Drive\Regularized-GradientTD\figures\figure_stepsize_sweep.png')