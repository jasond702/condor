"""
Future example problem, a bouncing ball problem! :)
Tests DAE with events.
Bouncing Ball acts like a DAESystem with no constraints (ODE)
At event, it becomes a DAE. So in reality it is ODE -> DAE -> ODE
"""

import matplotlib.pyplot as plt
import numpy as np

from condor.dae.model_templates import DAESystem
from condor.dae.dae_implementation import DAEAnalysisImplementation
from condor.backend.operators import pi


class BouncingBallProblem(DAESystem):
    coeff = parameter()
    g_const = parameter()

    height = state(initializer=20)
    velocity = state(initializer=0)

    initial_residual(height == 20)
    initial_residual(velocity == 0)

    shared_residual(dot[height] == velocity)
    shared_residual(dot[velocity] == -g_const)

    class Options:
        one_step = 1
        max_step = 0.01


class Bounce(BouncingBallProblem.Event):
    function = height
    update_residual(update[height] == 0)
    update_residual(update[velocity] == -coeff * velocity)
    # update_residual(update_dot[height] == update[velocity])
    # update_residual(update_dot[velocity] == -g_const)
    # coeff is 0 < coeff < 1

    # update[velocity] == -coeff * velocity
    # automate this ^, pass through or non changing state


class MaxHeight(BouncingBallProblem.Event):
    function = velocity


class EndEarly(BouncingBallProblem.Event):
    at_time = 2.5
    terminate = False


class BounceSim(BouncingBallProblem.DAEAnalysis):
    t0 = 0.0
    tf = 20.0


sim = BounceSim(coeff=0.9, g_const=9.81)

fig, ax = plt.subplots(2, 1)
ax[0].plot(sim.t, sim.state.height)
ax[0].set_xlabel("Time, $s$")
ax[0].set_ylabel("Height, $m$")
ax[0].grid(True)
ax[0].set_title("Bouncing Ball, Height vs. Time")
ax[1].plot(sim.t, sim.state.velocity)
ax[1].set_xlabel("Time, $s$")
ax[1].set_ylabel("Velocity, $m$")
ax[1].grid(True)
ax[1].set_title("Bouncing Ball, Velocity vs. Time")
plt.show()

breakpoint()
