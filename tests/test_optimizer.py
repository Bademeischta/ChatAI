import sys, os; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from src.optimizer import AdamOptimizer
from src.utils import Config


def test_adam_single_step():
    param = np.array([0.5, -0.1])
    grad = np.array([0.1, -0.02])
    adam = AdamOptimizer([param], lr=0.001)
    old_param = param.copy()
    adam.step([grad])
    assert not np.allclose(param, old_param)
