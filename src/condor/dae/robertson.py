import matplotlib.pyplot as plt
import numpy as np

from condor.dae.model_templates import DAESystem


# Robertson example
class RobertsonProblem(DAESystem):
    const1 = parameter()  # constant 1 = 0.04
    const2 = parameter()  # constant 2 = 10 x 10^4
    const3 = parameter()  # constant 3 = 3 x 10^7

    reactant_a = state(initializer=1.0)  # y0
    reactant_b = state(initializer=0)  # y1
    product_ab = state(initializer=0)  # y2

    t0 = 0.0
    tf = 10**6

    reactant_dot_a_ic = parameter()
    reactant_dot_b_ic = parameter()
    product_dot_ab_ic = parameter()

    initial_residual(dot[reactant_a] == reactant_dot_a_ic)
    initial_residual(dot[reactant_b] == reactant_dot_b_ic)
    initial_residual(dot[product_ab] == product_dot_ab_ic)

    shared_residual(
        dot[reactant_a] == -const1 * reactant_a + const2 * reactant_b * product_ab
    )
    shared_residual(
        dot[reactant_b]
        == const1 * reactant_a
        - const2 * reactant_b * product_ab
        - const3 * (reactant_b**2)
    )
    shared_residual(reactant_a + reactant_b + product_ab == 1)

    class Options:
        num_steps = 500
        logspace = True


sim1 = RobertsonProblem(
    const1=0.04,
    const2=1e4,
    const3=3e7,
    reactant_dot_a_ic=-0.04,
    reactant_dot_b_ic=0.04,
    product_dot_ab_ic=0,
)

sim1.state.reactant_b *= 1e4
plt.semilogx(sim1.t, sim1.state.reactant_a)
plt.semilogx(sim1.t, sim1.state.reactant_b)
plt.semilogx(sim1.t, sim1.state.product_ab)
plt.legend(["y0", "y1", "y2"])
plt.ylabel("Concentration, $c$")
plt.xlabel("Time, $t$")

if 0.0005625373157812227 in sim1:
    print("hi")
# 5.656176489796607
# 1359.7310543191368
# 11142.585755130078

# plt.scatter(
#     [0, sim2.state.x1[0], sim2.state.x2[0]],
#     [0, sim2.state.y1[0], sim2.state.y2[0]],
# )
plt.grid()
plt.show()
