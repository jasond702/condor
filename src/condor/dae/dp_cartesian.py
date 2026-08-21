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
    tf = 100
    x1 = state(initializer=0.7)
    y1 = state(initializer=-0.7)
    x2 = state(initializer=1.2)
    y2 = state(initializer=-1.5)
    vx1 = state(initializer=0)
    vy1 = state(initializer=0)
    vx2 = state(initializer=0)
    vy2 = state(initializer=0)
    lam1 = state(initializer=0)
    lam2 = state(initializer=0)

    initial_residual(L1_length_meter * np.sin(theta1 * (pi / 180)) == x1)
    initial_residual(y1 == -L1_length_meter * np.cos(theta1 * (pi / 180)))
    initial_residual(x2 == x1 + L2_length_meter * np.sin(theta2 * (pi / 180)))
    initial_residual(y2 == y1 - L2_length_meter * np.cos(theta2 * (pi / 180)))
    initial_residual(vx1 == 0)
    initial_residual(vy1 == 0)
    initial_residual(vx2 == 0)
    initial_residual(vy2 == 0)
    initial_residual(dot[lam1] == 0)
    initial_residual(dot[lam2] == 0)

    shared_residual(dot[x1] == vx1)
    shared_residual(dot[y1] == vy1)
    shared_residual(dot[x2] == vx2)
    shared_residual(dot[y2] == vy2)

    shared_residual(m1_mass_kg * dot[vx1] == 2 * lam1 * x1 - 2 * lam2 * (x2 - x1))
    shared_residual(
        m1_mass_kg * dot[vy1]
        == -m1_mass_kg * g_const + 2 * lam1 * y1 - 2 * lam2 * (y2 - y1)
    )
    shared_residual(m2_mass_kg * dot[vx2] == 2 * lam2 * (x2 - x1))
    shared_residual(
        m2_mass_kg * dot[vy2] == -m2_mass_kg * g_const + 2 * lam2 * (y2 - y1)
    )

    shared_residual(x1 * dot[vx1] + y1 * dot[vy1] + vx1**2 + vy1**2 == 0)
    shared_residual(
        (x2 - x1) * (dot[vx2] - dot[vx1])
        + (y2 - y1) * (dot[vy2] - dot[vy1])
        + (vx2 - vx1) ** 2
        + (vy2 - vy1) ** 2
        == 0
    )

    class Options:
        num_steps = 10000
        linspace = True
        # solver_kind =


sim2 = DoublePendulumProblem(
    L1_length_meter=1.0,
    L2_length_meter=1.0,
    g_const=9.81,
    m1_mass_kg=1.0,
    m2_mass_kg=1.0,
    theta1=45,
    theta2=30,
)

plt.plot(sim2.state.x1, sim2.state.y1)
plt.plot(sim2.state.x2, sim2.state.y2)
plt.legend(["Mass 1", "Mass 2"])
plt.xlabel("x Position, $m$")
plt.ylabel("y Position, $m$")
plt.grid()
plt.title("x vs. y, Condor")
# plt.scatter(
#     [0, sim2.state.x1[0], sim2.state.x2[0]],
#     [0, sim2.state.y1[0], sim2.state.y2[0]],
# )
plt.axis("equal")
plt.show()
