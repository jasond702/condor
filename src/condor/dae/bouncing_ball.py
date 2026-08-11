"""
Future example problem, a bouncing ball problem! :)
Tests DAE with events.
Bouncing Ball acts like a DAESystem with no constraints (ODE)
At event, it becomes a DAE. So in reality it is ODE -> DAE -> ODE
"""

import matplotlib.pyplot as plt
import numpy as np

from condor.dae.model_templates import DAESystem
from condor.backend.operators import pi


class BouncingBallProblem(DAESystem):
    coeff = parameter()
    g_const = parameter()

    height = state()
    velocity = state()

    initial_residual(height == 5)
    initial_residual(velocity == 0)

    shared_residual(dot[velocity] == -g_const)
    shared_residual(dot[height] == velocity)


class Bounce(BouncingBallProblem.DAEEvent):
    function = height
    # update[velocity] == -coeff * velocity
    update_residual(update[velocity] == -coeff * velocity)
    update_residual(update[height] == height)
    # automate this ^, pass through or non changing state


sim = BouncingBallProblem(coeff=0.5, g_const=9.81)
