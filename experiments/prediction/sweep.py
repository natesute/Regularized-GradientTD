import numpy as np
import pickle

# source found at: https://github.com/andnp/RlGlue
from RlGlue import RlGlue
from utils.Collector import Collector
from utils.rl_glue import RlGlueCompatWrapper
from utils.errors import buildRMSPBE
from utils.weighting import WEIGHTINGS
from utils.config import PROBLEMS, STEPSIZES

from agents.TD import TD
from agents.TDC import TDC
from agents.HTD import HTD
from agents.GTD2 import GTD2
from agents.TDRC import TDRC
from agents.Vtrace import Vtrace

RUNS = 20
LEARNERS = [GTD2]

# -----------------------------------
# Collect Data for Stepsize Sweep
# -----------------------------------

collector = Collector()

# dictionary to store stepsizes with minimum RMSPBE
min_stepsizes = {}

for run in range(RUNS):
    for Learner in LEARNERS:
        for weighting_num, weighting in enumerate(WEIGHTINGS):
            for problem in PROBLEMS:
                env_name = problem['env'].__name__
                rep_name = problem['representation'].__name__
                
                mean_rmspbes = np.zeros(len(STEPSIZES))
                for i, stepsize in enumerate(STEPSIZES):
                    # Reproducibility and reset
                    np.random.seed(run)

                    # Environment and representation
                    Env = problem['env']
                    env = Env()
                    target = problem['target']
                    Rep = problem['representation']
                    rep = Rep(weighting=weighting)

                    data_key = f"{env_name}-{rep_name}-{weighting.__name__}-{stepsize}"

                    print(f"Run {run}, Env: {env_name}, Rep: {rep_name}, Weighting: {weighting.__name__}, Stepsize: {stepsize}")

                    X, P, R, D = env.getXPRD(target, rep)
                    RMSPBE = buildRMSPBE(X, P, R, D, problem['gamma'])

                    # Learner with current stepsize
                    learner = Learner(rep.features(), {
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
                            # print("Collecting data for", env_name, rep_name, weighting.__name__, stepsize, "at step", step, "RMSPBE:", rmspbe)
                            collector.collect(data_key, rmspbe)

                    # tell the data collector we're done collecting data for this env/learner/rep/weighting/stepsize combination
                    collector.reset()
                    # print("Stats for", data_key, ":", collector.getStats(data_key))
                    # print("mean_rmspbes", mean_rmspbes)
                    mean_rmspbes[i] = collector.getStats(data_key)[0].mean()

                # get stepsize with min RMSPBE (and not NaN)
                min_stepsize_index = np.nanargmin(mean_rmspbes)
                min_stepsize = STEPSIZES[min_stepsize_index]

                print(f"Min RMSPBE at stepsize: {min_stepsize} for {weighting.__name__} weighting")

                # store the min stepsize for this env/rep/weighting combination
                min_stepsizes[(weighting.__name__)] = min_stepsize

with open('sweep.pkl', 'wb') as f:
    pickle.dump(collector, f)

# pickle the min_stepsizes dictionary for use in training
with open('min_stepsizes.pkl', 'wb') as f:
    pickle.dump(min_stepsizes, f)