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
from condor.implementations.utils import options_to_kwargs
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
        self.options_dict = options_to_kwargs(self.model)
        self.p = self.model.parameter.flatten()

        self.differential_state = self.model.differential_state.flatten()
        self.algebraic_state = self.model.algebraic_state.flatten()
        self.dot = self.model.dot.flatten()

        self.initial_residual = self.model.initial_residual.flatten()
        self.residual = self.model.residual.flatten()
        self.output = self.model.output.flatten()
        self.final_time = self.model.tf
        self.start_time = self.model.t0

        self.state_count = (
            self.differential_state.shape[0] + self.algebraic_state.shape[0]
        )
        self.dot_count = self.dot.shape[0]
        self.total_count = self.state_count + self.dot_count
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
            # create dot for the algebraic state?
            # this would get rid of the append for ICs
            input_dict = residual_dict.as_("backend_repr")
            existing_vars = []
            for index, elem in enumerate(self.model.initial_residual):
                residual(
                    substitute(elem.backend_repr, input_dict),
                    name=f"residual_{index}",
                )
                try:
                    existing_var = residual._elements[index].backend_repr.dep(0).name()
                except RuntimeError:
                    existing_var = residual._elements[index].backend_repr.dep(1).name()
                existing_vars.append(existing_var)
            breakpoint()

            # algebraic states usually never have an initial condition
            # (especially their derivative)
            if len(residual) < len(variable):
                residual_len = len(residual)
                for _elem in variable:
                    if _elem.backend_repr.name() in existing_vars:
                        pass
                    else:
                        residual(
                            _elem.backend_repr == 0, name=f"residual_{residual_len}"
                        )
                        residual_len += 1
            breakpoint()

        self.initial_conditions = DAEInitialConditionSolve(**model_instance.parameter)
        # breakpoint()
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

        self.dae_analysis_soln = DAEAnalysis(
            initial_conditions=self.initial_conditions,
            final_time=self.final_time,
            start_time=self.start_time,
            state_count=self.state_count,
            dot_count=self.dot_count,
            count_diff=self.count_diff,
            residual_func=self.residual_func,
            **self.options_dict,
        )

        self(model_instance)

    def __call__(self, model_instance):
        soln = self.dae_analysis_soln()
        print(soln)

        differential_state_soln = np.stack(
            [soln.y[:, x] for x in range(self.dot_count)]
        )
        algebraic_state_soln = np.stack(
            [soln.y[:, x] for x in range(self.dot_count, self.state_count)]
        )
        # should we bind the dot of algebraic states? are they interesting to look at?
        dot_soln = np.stack([soln.yp[:, x] for x in range(self.dot_count)])

        model_instance.t = soln.t
        model_instance.bind_field(
            model_instance.__class__.differential_state.wrap(differential_state_soln)
        )
        model_instance.bind_field(
            model_instance.__class__.algebraic_state.wrap(algebraic_state_soln)
        )
        model_instance.bind_field(model_instance.__class__.dot.wrap(dot_soln))


class DAEAnalysis:
    # moving towards wrapping pysundae, might need something like class System
    def __init__(
        self,
        initial_conditions,
        final_time,
        start_time,
        state_count,
        dot_count,
        count_diff,
        residual_func,
        **analysis_options,
    ):
        self.state_count = state_count
        self.dot_count = dot_count
        self.residual_func = residual_func
        self.initial_conditions = initial_conditions

        self.initial_state = initial_conditions.variable.flatten()[:state_count]
        self.initial_dot = initial_conditions.variable.flatten()[state_count:]
        breakpoint()
        # i feel like this will eventually throw an error (think it over, maybe not?)

        # dae folder -> model templates, model types, implementations (possibly),
        # solvers.py (has this DAEAnalysis) building towards the wrapper,
        # sub model templates for events (similar to class Event())
        # instead of update something like reinitialize residual
        # think about API for shared residual

        if len(self.initial_state) < len(self.initial_dot):
            self.initial_state = np.append(self.initial_state, [0] * count_diff)
        elif len(self.initial_state) > len(self.initial_dot):
            self.initial_dot = np.append(self.initial_dot, [0] * count_diff)

        # probably could add like in trajectory analysis to take out the key and value
        num_steps = analysis_options.pop("num_steps")

        if "linspace" in analysis_options and analysis_options.pop("linspace"):
            linspace = True
        elif "logspace" in analysis_options and analysis_options.pop("logspace"):
            logspace = True
            linspace = False
        else:
            linspace = True  # default
        # i feel like i can combine these
        if linspace:
            self.tspan = np.linspace(start_time, final_time, num_steps)
        elif logspace:
            if start_time == 0 and final_time > 100:
                self.tspan = np.logspace(-6, np.log10(final_time), num_steps)
            else:
                self.tspan = np.logspace(start_time, final_time, num_steps)

    # look at trajectoryAnalysis for options and pulling out t0, tf for tspan
    # ignore events stuff for now

    # Plan Outline/Task List:
    # DAEAnalysis -> TrajectoryAnalysis (look like) (DID?)
    # use options to set num steps between t0 and tf for tspan,
    # also flag to lin or logspace (DID?)
    # Solver look like sgm_solver -> so it can handle time updates for event handling
    # event stuff
    # sgm for DAEs

    def __call__(self):
        def residualfunction(t, y, yp, res):
            res[:, None] = self.residual_func(
                y[: self.dot_count],
                y[self.dot_count :],
                yp[: self.dot_count],
                self.initial_conditions.parameter.flatten(),
            )

        if all(x == 0 for x in self.initial_state):
            solver = ida.IDA(
                residualfunction,
                atol=1e-8,
                algebraic_idx=list(range(self.state_count)[self.dot_count :]),
                calc_initcond="y0",
            )
        elif all(x == 0 for x in self.initial_dot):
            solver = ida.IDA(
                residualfunction,
                atol=1e-8,
                algebraic_idx=list(range(self.state_count)[self.dot_count :]),
                calc_initcond="yp0",
            )
        else:
            solver = ida.IDA(
                residualfunction,
                atol=1e-8,
                algebraic_idx=list(range(self.state_count)[self.dot_count :]),
            )

        soln = solver.solve(self.tspan, self.initial_state, self.initial_dot)
        return soln


# model type
class DAESystemType(ModelType):
    @classmethod
    def process_placeholders(cls, new_cls, attrs):
        super().process_placeholders(new_cls, attrs)
        for elem in new_cls.residual:
            process_relational_element(elem)

        # do same for initial_residual (process_relational_element)

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
    # initial_residual = SharedField(Direction.internal)
    output = AssignedField(Direction.output)


# TODO: add subtemplate to handle events
# submodel type for events
# class DAEEventType(SubmodelType):
# @classmethod

# submodel template for events
# class DAEEvent(SubmodelTemplate, model_metaclass=DAEEventType, primary=DAESystem):


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

    initial_residual(L1_length_meter * np.sin(theta1 * (pi / 180)) == x1)
    initial_residual(y1 == -L1_length_meter * np.cos(theta1 * (pi / 180)))
    # replace with x^2 + y^2 = L^2 and arctan = theta1
    initial_residual(x2 == x1 + L2_length_meter * np.sin(theta2 * (pi / 180)))
    initial_residual(y2 == y1 - L2_length_meter * np.cos(theta2 * (pi / 180)))
    initial_residual(vy2 == 0.5)
    initial_residual(vx2 == 0.5)
    # helper function?
    # TODO: fix initial residual stuff
    # shared residual <- new fieldtype (straight from field?)
    # __init__ list of fields it's copying to and
    # __call__(*args, **kwargs) iterate through fields and calls them
    # initial theta1 dot and theta2 dot

    # something like class CombinedFieldResidual

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
    theta2=45,
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
breakpoint()

# bouncing ball problem :)
