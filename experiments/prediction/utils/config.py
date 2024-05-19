import numpy as np

from environments.RandomWalk import RandomWalk, TabularRep, DependentRep, InvertedRep
from environments.Boyan import Boyan, BoyanRep
from environments.Baird import Baird, BairdRep

from utils.policies import actionArrayToPolicy, matrixToPolicy

STEPSIZES = [2**(i) for i in range(-6, 7)]

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
            'GTD2': [0.03125, 0.015625, 0.03125],
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
            'GTD2': [0.0625, 0.03125, 0.0625],
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
            'GTD2': [0.125, 0.015625, 0.0625],
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
            'GTD2': [0.5, 0.0078125, 0.03125],
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
            'GTD2': [0.00390625, 0.00390625, 0.00390625],
            'HTD': 0.00390625,
            'Vtrace': 0.00390625,
        }
    },
]