import matplotlib.pyplot as plt
import numpy as np

from condor.dae.model_templates import DAESystem


# Robertson example
class RobertsonProblem(DAESystem):
    const1 = parameter()  # constant 1 = 0.04
    const2 = parameter()  # constant 2 = 10 x 10^4
    const3 = parameter()  # constant 3 = 3 x 10^7
    product_ab = algebraic_state()
    reactant_a = differential_state()
    reactant_b = differential_state()
    t0 = 0
    tf = 1000000.0

    reactant_a_ic = parameter()
    reactant_b_ic = parameter()
    product_ab_ic = parameter()
    reactant_dot_a_ic = parameter()
    reactant_dot_b_ic = parameter()
    product_dot_ab_ic = parameter()

    initial_residual(reactant_a == reactant_a_ic)
    initial_residual(reactant_b == reactant_b_ic)
    initial_residual(product_ab == product_ab_ic)
    initial_residual(dot[reactant_a] == reactant_dot_a_ic)
    initial_residual(dot[reactant_b] == reactant_dot_b_ic)

    residual(dot[reactant_a] == -const1 * reactant_a + const2 * reactant_b * product_ab)
    residual(
        dot[reactant_b]
        == const1 * reactant_a
        - const2 * reactant_b * product_ab
        - const3 * (reactant_b**2)
    )
    residual(reactant_a + reactant_b + product_ab == 1)

    class Options:
        num_steps = 50
        logspace = True


sim1 = RobertsonProblem(
    const1=0.04,
    const2=1e4,
    const3=3e7,
    reactant_a_ic=1,
    reactant_b_ic=0,
    product_ab_ic=0,
    reactant_dot_a_ic=-0.04,
    reactant_dot_b_ic=0.04,
    product_dot_ab_ic=0,
)

sim1.differential_state.reactant_b *= 1e4
plt.semilogx(sim1.t, sim1.differential_state.reactant_a)
plt.semilogx(sim1.t, sim1.differential_state.reactant_b)
plt.semilogx(sim1.t, sim1.algebraic_state.product_ab)
plt.legend(["y0", "y1", "y2"])
plt.ylabel("Concentration, $c$")
plt.xlabel("Time, $t$")
plt.grid()
plt.show()
