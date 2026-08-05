import matplotlib.pyplot as plt
import numpy as np

from condor.dae.model_templates import DAESystem
from condor.backend.operators import pi


class DoublePendulumProblem(DAESystem):
    L1_length_meter = parameter()
    L2_length_meter = parameter()

    g_const = parameter()

    m1_mass_kg = parameter()
    m2_mass_kg = parameter()

    theta1 = parameter()
    theta2 = parameter()

    t0 = 0
    tf = 20

    x1 = differential_state(initializer=0.7)
    y1 = differential_state(initializer=-0.7)
    x2 = differential_state(initializer=1.2)
    y2 = differential_state(initializer=-1.5)
    vx1 = differential_state(initializer=0)
    vy1 = differential_state(initializer=0)
    vx2 = differential_state(initializer=0)
    vy2 = differential_state(initializer=0)
    lam1 = algebraic_state(initializer=0)
    lam2 = algebraic_state(initializer=0)

    initial_residual(x1**2 + y1**2 == L1_length_meter**2)
    initial_residual((x2 - x1) ** 2 + (y2 - y1) ** 2 == L2_length_meter**2)
    initial_residual(np.tan(theta1 * (pi / 180)) == -x1 / y1)
    initial_residual(np.tan(theta2 * (pi / 180)) == (x2 - x1) / (-(y2 - y1)))

    # initial_residual(np.atan2(x1, y1) == theta1 * (pi / 180))
    # initial_residual(np.atan2((x2 - x1), -(y2 - y1)) == theta2 * (pi / 180))
    # atan and atan2 will work but the user has to figure out how to best represent it

    initial_residual(vx1 == 0)
    initial_residual(vy1 == 0)
    initial_residual(vx2 == 0)
    initial_residual(vy2 == 0)
    initial_residual(lam1 == 0)
    initial_residual(lam2 == 0)
    initial_residual(dot[x1] == 0)
    initial_residual(dot[y1] == 0)
    initial_residual(dot[x2] == 0)
    initial_residual(dot[y2] == 0)
    initial_residual(dot[vx1] == 0)
    initial_residual(dot[vy1] == 0)
    initial_residual(dot[vx2] == 0)
    initial_residual(dot[vy2] == 0)

    residual(dot[x1] == vx1)
    residual(dot[y1] == vy1)
    residual(dot[x2] == vx2)
    residual(dot[y2] == vy2)

    residual(m1_mass_kg * dot[vx1] == 2 * lam1 * x1 - 2 * lam2 * (x2 - x1))
    residual(
        m1_mass_kg * dot[vy1]
        == -m1_mass_kg * g_const + 2 * lam1 * y1 - 2 * lam2 * (y2 - y1)
    )

    residual(m2_mass_kg * dot[vx2] == 2 * lam2 * (x2 - x1))
    residual(m2_mass_kg * dot[vy2] == -m2_mass_kg * g_const + 2 * lam2 * (y2 - y1))

    residual(x1 * dot[vx1] + y1 * dot[vy1] + vx1**2 + vy1**2 == 0)
    residual(
        (x2 - x1) * (dot[vx2] - dot[vx1])
        + (y2 - y1) * (dot[vy2] - dot[vy1])
        + (vx2 - vx1) ** 2
        + (vy2 - vy1) ** 2
        == 0
    )

    class Options:
        num_steps = 100
        linspace = True


sim2 = DoublePendulumProblem(
    L1_length_meter=1.0,
    L2_length_meter=1.0,
    g_const=9.81,
    m1_mass_kg=1.0,
    m2_mass_kg=1.0,
    theta1=45,
    theta2=30,
)

plt.plot(sim2.differential_state.x1, sim2.differential_state.y1)
plt.plot(sim2.differential_state.x2, sim2.differential_state.y2)
plt.legend(["Mass 1", "Mass 2"])
plt.xlabel("x Position, $m$")
plt.ylabel("y Position, $m$")
plt.grid()
plt.scatter(
    [0, sim2.differential_state.x1[0], sim2.differential_state.x2[0]],
    [0, sim2.differential_state.y1[0], sim2.differential_state.y2[0]],
)
plt.axis("equal")
plt.show()
