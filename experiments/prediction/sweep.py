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

for run in range(RUNS):
    for problem in PROBLEMS:
        for Learner in LEARNERS:
            for weighting_num, weighting in enumerate(WEIGHTINGS):
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
                    rep = Rep(weighting=weighting)

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
                            collector.collect(f"{env_name}-{rep_name}-{weighting.__name__}-{stepsize}", rmspbe)

                    # tell the data collector we're done collecting data for this env/learner/rep/weighting/stepsize combination
                    collector.reset()
with open('sweep.pkl', 'wb') as f:
    pickle.dump(collector, f)