"""
Typical Predator Prey problem. System of ODEs.
Used to test if ODEs can be solved with DAE solver.
ODEs are just DAEs with no algebraic constraint.
Update: Works! :)
"""

import matplotlib.pyplot as plt
import numpy as np

from condor.dae.model_templates import DAESystem
from condor.backend.operators import pi


class PredatorPreyProblem(DAESystem):
    const1 = parameter()
    const2 = parameter()
    const3 = parameter()
    const4 = parameter()
    t0 = 0.0
    tf = 10.0

    y0 = differential_state(initializer=0.0)
    y1 = differential_state(initializer=0.0)
    initial_residual(y0 == 1.0)
    initial_residual(y1 == 1.0)
    initial_residual(dot[y0] == 0.0)
    initial_residual(dot[y1] == 0.0)

    residual(dot[y0] == const1 * y0 - const2 * y0 * y1)
    residual(dot[y1] == -const3 * y1 + const4 * y0 * y1)

    class Options:
        num_steps = 50


sim = PredatorPreyProblem(const1=1.5, const2=1.0, const3=3.0, const4=1.0)

fig, ax = plt.subplots(1, 2)
ax[0].plot(sim.t, sim.differential_state.y0)
ax[0].plot(sim.t, sim.differential_state.y1)
ax[0].legend(["Prey", "Predator"])
ax[0].set_xlabel("Prey")
ax[0].set_ylabel("Predator")
ax[0].set_title("Population vs. Time")
ax[0].grid()

ax[1].plot(sim.differential_state.y0, sim.differential_state.y1)
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("Population")
ax[1].set_title("Phase Portrait")
ax[1].grid()

plt.show()
