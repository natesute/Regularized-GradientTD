import numpy as np
import pickle

# source found at: https://github.com/andnp/RlGlue
from RlGlue import RlGlue
from utils.Collector import Collector
from utils.rl_glue import RlGlueCompatWrapper
from utils.errors import buildRMSPBE
from utils.weighting import WEIGHTINGS
from utils.config import PROBLEMS

from agents.TD import TD
from agents.TDC import TDC
from agents.HTD import HTD
from agents.GTD2 import GTD2
from agents.TDRC import TDRC
from agents.Vtrace import Vtrace

# --------------------------------
# Set up parameters for experiment
# --------------------------------

RUNS = 20
LEARNERS = [GTD2]#, TDC, Vtrace, HTD, TD, TDRC]

# load the precomputed optimal stepsizes
with open('min_stepsizes.pkl', 'rb') as f:
    min_stepsizes = pickle.load(f)

# -----------------------------------
# Collect the data for the experiment
# -----------------------------------

# a convenience object to store data collected during runs
collector = Collector()

for run in range(RUNS):
    for problem in PROBLEMS:
        for Learner in LEARNERS:
            for weighting_num, weighting in enumerate(WEIGHTINGS):
                # for reproducibility, set the random seed for each run
                # also reset the seed for each learner, so we guarantee each sees the same data
                np.random.seed(run)

                # build a new instance of the environment each time
                # just to be sure we don't bleed one learner into the next
                Env = problem['env']
                env = Env()

                target = problem['target']
                behavior = problem['behavior']

                Rep = problem['representation']
                rep = Rep(weighting=weighting)

                print(run, Env.__name__, Rep.__name__, "Weighting:", weighting.__name__, Learner.__name__)

                # build the X, P, R, and D matrices for computing RMSPBE
                X, P, R, D = env.getXPRD(target, rep)
                RMSPBE = buildRMSPBE(X, P, R, D, problem['gamma'])

                # get the precomputed stepsize (alpha)
                alpha = min_stepsizes[(Env.__name__, Rep.__name__, weighting.__name__)]

                # build a new instance of the learning algorithm
                learner = Learner(rep.features(), {
                    'gamma': problem['gamma'],
                    'alpha': alpha,
                    'beta': 1,
                })

                # build an "agent" which selects actions according to the behavior
                # and tries to estimate according to the target policy
                agent = RlGlueCompatWrapper(learner, behavior, target, rep.encode)

                # for Baird's counter-example, set the initial value function manually
                if problem.get('starting_condition') is not None:
                    learner.w = problem['starting_condition'].copy()

                # build the experiment runner
                # ties together the agent and environment
                # and allows executing the agent-environment interface from Sutton-Barto
                glue = RlGlue(agent, env)

                # start the episode (env produces a state then agent produces an action)
                glue.start()
                for step in range(problem['steps']):
                    # interface sends action to env and produces a next-state and reward
                    # then sends the next-state and reward to the agent to make an update
                    _, _, _, terminal = glue.step()

                    # when we hit a terminal state, start a new episode
                    if terminal:
                        glue.start()

                    # evaluate the RMPSBE
                    # subsample to reduce computational cost
                    if step % 100 == 0:
                        print("step:", step)
                        w = learner.getWeights()
                        rmspbe = RMSPBE(w)
                        print('rmspbe:', rmspbe)

                        #  create a unique key to store the data for this env/representation/agent tuple
                        data_key = f'{Env.__name__}-{Rep.__name__}-{weighting.__name__}-{Learner.__name__}'
                        # store the data in the "collector" until we need it for plotting
                        collector.collect(data_key, rmspbe)

                # tell the data collector we're done collecting data for this env/learner/rep/weighting combination
                collector.reset()

with open('training.pkl', 'wb') as f:
    pickle.dump(collector, f)