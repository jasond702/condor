import logging  # don't know if i need this

import matplotlib.pyplot as plt
import numpy as np  # probably need this
from sksundae import ida
from utils import ElementMap

from condor import (
    AlgebraicSystem,
)
from condor.backend import (  # probably need these
    expression_to_operator,
    process_relational_element,
)

# from condor.backend.operators import substitute  # probably need
from condor.backend.operators import pi, substitute
from condor.fields import (  # keep all fields?
    AssignedField,
    Direction,
    FreeAssignedField,
    FreeField,
    FreeMatchedField,
)
from condor.models import (  # keep these
    ModelTemplate,
    ModelType,
)

log = logging.getLogger(__name__)

# most imports are based on contrib.py


class DAEAnalysisImplementation:
    def __init__(self, model_instance):
        self.extra_args = dict(
            cse=True,
        )
        self.model = model_instance.__class__

        self.p = self.model.parameter.flatten()

        self.differential_state = self.model.differential_state.flatten()
        self.algebraic_state = self.model.algebraic_state.flatten()
        self.dot = self.model.dot.flatten()

        self.initial_residual = self.model.initial_residual.flatten()
        self.residual = self.model.residual.flatten()
        self.output = self.model.output.flatten()
        breakpoint()

        self.state_count = (
            self.differential_state.shape[0] + self.algebraic_state.shape[0]
        )
        self.dot_count = self.dot.shape[0]
        self.count_diff = self.state_count - self.dot_count

        # solve for the initial conditions from the initial residuals
        class DAEInitialConditionSolve(AlgebraicSystem):
            residual_dict = ElementMap()
            for elem in self.model.parameter:
                residual_dict[elem] = parameter(
                    name=elem.name,
                    shape=elem.shape,
                )
            for elem in self.model.differential_state:
                residual_dict[elem] = variable(
                    name=elem.name,
                    shape=elem.shape,
                )
            for elem in self.model.algebraic_state:
                residual_dict[elem] = variable(
                    name=elem.name,
                    shape=elem.shape,
                )
            for elem in self.model.dot:
                backend_name = f"{elem.name}_dot"
                residual_dict[elem] = variable(
                    name=backend_name,
                    shape=elem.shape,
                )

            input_dict = residual_dict.as_("backend_repr")
            for elem in self.model.initial_residual:
                residual(substitute(elem.backend_repr, input_dict))

            breakpoint()

        self.initial_conditions = DAEInitialConditionSolve(**model_instance.parameter)
        breakpoint()

        # create 2 vectors that have ICs for the states and dots
        self.initial_state = []
        self.initial_dot = []

        for elem in self.model.differential_state:
            self.initial_state.append(
                self.initial_conditions.variable[elem.name].item()
            )
            dot_name = f"{elem.name}_dot"
            self.initial_dot.append(self.initial_conditions.variable[dot_name].item())
        for elem in self.model.algebraic_state:
            self.initial_state.append(
                self.initial_conditions.variable[elem.name].item()
            )

        if len(self.initial_state) < len(self.initial_dot):
            self.initial_state.extend([0] * self.count_diff)
        elif len(self.initial_state) > len(self.initial_dot):
            self.initial_dot.extend([0] * self.count_diff)

        self.residual_vars = [
            self.differential_state,
            self.algebraic_state,
            self.dot,
            self.p,
        ]
        self.residual_func = expression_to_operator(
            self.residual_vars,
            self.residual,
            f"{self.model.__name__}_residual",
        )
        breakpoint()

        def residualfunction(t, y, yp, res):
            res[:, None] = self.residual_func(
                y[: self.differential_state.shape[0]],
                y[self.differential_state.shape[0] :],
                yp[: self.dot.shape[0]],
                self.initial_conditions.parameter.flatten(),
            )
            # breakpoint()

        # TODO: some how pick linspace or logspace or pick one and
        # let the solver deal with it
        if self.state_count == 3:
            tspan = np.logspace(-6, 6, 500)
        elif self.state_count == 10:
            tspan = np.linspace(0, 20, 100)

        # algebraic_idx = [idx for idx in range(self.state_count) if idx == 1]
        breakpoint()
        if all(x == 0 for x in self.initial_state):
            solver = ida.IDA(
                residualfunction,
                atol=1e-8,
                algebraic_idx=list(
                    range(self.state_count)[self.differential_state.shape[0] :]
                ),
                calc_initcond="y0",
            )
        elif all(x == 0 for x in self.initial_dot):
            solver = ida.IDA(
                residualfunction,
                atol=1e-8,
                algebraic_idx=list(
                    range(self.state_count)[self.differential_state.shape[0] :]
                ),
                calc_initcond="yp0",
            )
        else:
            solver = ida.IDA(
                residualfunction,
                atol=1e-8,
                algebraic_idx=list(
                    range(self.state_count)[self.differential_state.shape[0] :]
                ),
            )

        self.soln = solver.solve(tspan, self.initial_state, self.initial_dot)
        breakpoint()
        self(model_instance)

    # TODO: bind and wrap output for user to handle?
    def __call__(self, model_instance):
        soln = self.soln
        print(soln)
        if soln.y.shape[1] == 3:
            soln.y[:, 1] *= 1e4  # scale y1 values for plotting
            plt.semilogx(soln.t, soln.y)
            plt.legend(["y0", "y1", "y2"])
            plt.xlabel("Time (s), $t$")
            plt.ylabel("Concentration, $c$")
            plt.grid()
            plt.show()
        elif soln.y.shape[1] == 10:
            x1 = soln.y[:, 0]
            y1 = soln.y[:, 1]
            x2 = soln.y[:, 2]
            y2 = soln.y[:, 3]
            plt.plot(x1, y1)
            plt.plot(x2, y2)
            plt.legend(["Mass 1", "Mass 2"])
            plt.xlabel("x Position, $m$")
            plt.ylabel("y Position, $m$")
            plt.grid()
            plt.scatter([0, x1[0], x2[0]], [0, y1[0], y2[0]])
            plt.axis("equal")
            plt.show()
        else:
            print("Not a Robertson or Double Pendulum problem.")


# model type
class DAESystemType(ModelType):
    @classmethod
    def process_placeholders(cls, new_cls, attrs):
        super().process_placeholders(new_cls, attrs)
        for elem in new_cls.residual:
            process_relational_element(elem)

    implementation = DAEAnalysisImplementation


# model template
class DAESystem(ModelTemplate, model_metaclass=DAESystemType):
    t = placeholder(default=None)
    t0 = placeholder(default=None)
    tf = placeholder(default=np.inf)

    parameter = FreeField()
    residual = FreeAssignedField(Direction.internal)

    differential_state = FreeField(Direction.internal)
    algebraic_state = FreeField(Direction.internal)
    dot = FreeMatchedField(differential_state)

    initial_residual = FreeAssignedField(Direction.internal)
    output = AssignedField(Direction.output)


# TODO: add subtemplate to handle events
# submodel type for events
# class DAEEventType(SubmodelType):
# @classmethod

# submodel template for events
# class DAEEvent(SubmodelTemplate, model_metaclass=DAEEventType, primary=DAESystem):

# TODO: pysundae IDA requires a t0, tf, and an n amount of points


# Robertson example
class RobertsonProblem(DAESystem):
    const1 = parameter()  # constant 1 = 0.04
    const2 = parameter()  # constant 2 = 10 x 10^4
    const3 = parameter()  # constant 3 = 3 x 10^7
    product_ab = algebraic_state()
    reactant_a = differential_state()
    reactant_b = differential_state()

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


RobertsonProblem(
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


class DoublePendulumProblem(DAESystem):
    L1_length_meter = parameter()
    L2_length_meter = parameter()

    g_const = parameter()

    m1_mass_kg = parameter()
    m2_mass_kg = parameter()

    theta1 = parameter()
    theta2 = parameter()

    x1 = differential_state()
    y1 = differential_state()
    x2 = differential_state()
    y2 = differential_state()
    vx1 = differential_state()
    vy1 = differential_state()
    vx2 = differential_state()
    vy2 = differential_state()
    lam1 = algebraic_state()
    lam2 = algebraic_state()

    initial_residual(x1 == L1_length_meter * np.sin(theta1 * (pi / 180)))
    initial_residual(y1 == -L1_length_meter * np.cos(theta1 * (pi / 180)))
    initial_residual(x2 == x1 + L2_length_meter * np.sin(theta2 * (pi / 180)))
    initial_residual(y2 == y1 - L2_length_meter * np.cos(theta2 * (pi / 180)))
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


DoublePendulumProblem(
    L1_length_meter=1.0,
    L2_length_meter=1.0,
    g_const=9.81,
    m1_mass_kg=1.0,
    m2_mass_kg=1.0,
    theta1=45,
    theta2=30,
)
