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
from condor.backend.operators import substitute  # probably need
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

        self.initial_conditions = DAEInitialConditionSolve(**model_instance.parameter)

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
            self.initial_state.append(0)
        elif len(self.initial_state) > len(self.initial_dot):
            self.initial_dot.append(0)

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

        def residualfunction(t, y, yp, res):
            res[:, None] = self.residual_func(
                y[:2], y[2], yp[:2], self.initial_conditions.parameter.flatten()
            )

        # TODO: only works for RobertsonProblem for now, will update such that
        # it works fro any DAE system
        tspan = np.logspace(-6, 6, 50)
        solver = ida.IDA(residualfunction, atol=1e-8, algebraic_idx=[2])
        self.soln = solver.solve(tspan, self.initial_state, self.initial_dot)

        self(model_instance)

    # TODO: __call__ will return pysundae solution
    def __call__(self, model_instance):
        soln = self.soln
        print(soln)
        # model_instance._soln = soln
        soln.y[:, 1] *= 1e4  # scale y1 values for plotting
        plt.semilogx(soln.t, soln.y)
        plt.legend(["y0", "y1", "y2"])
        plt.xlabel("Time (s), $t$")
        plt.ylabel("Concentration, $c$")
        plt.show()
        # breakpoint()
        # breakpoint()
        # print(soln)
        # soln.y[:,1] *= 1e4 #scale y1 values for plotting
        # plt.semilogx(soln.t, soln.y)
        # plt.legend(["y0", "y1", "y2"])
        # plt.xlabel("Time (s), $t$")
        # plt.ylabel("Concentration, $c$")


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
    initial_residual = FreeAssignedField(Direction.internal)
    dot = FreeMatchedField(differential_state)
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
