from condor.backend.operators import substitute, concat, inf, jacobian
from condor.utils import ElementMap

from condor.backend import expression_to_operator, symbol_class
from condor import AlgebraicSystem
from condor.dae.solvers_adjoint import (
    DAESystemAnalysis,
    TimeGeneratorFromSlices,
    NextTimeFromSlice,
)
from condor.implementations.utils import options_to_kwargs
import numpy as np
from condor.fields import BaseElement
from casadi import MX, sum2


def get_state_setter(field, signature, on_field=None, subs=None):
    expr = field.flatten(on_field)
    if subs is not None:
        expr = substitute(expr, subs)
    func = expression_to_operator(
        signature,
        expr,
        f"{field._model_name}_{field._matched_to._name}_{field._name}",
    )
    func.expr = expr
    return func


def isnan(x):
    return isinstance(x, float) and np.isnan(x)


class DAEAnalysisImplementation:
    def __init__(self, model_instance):
        self.extra_args = dict(
            cse=True,
        )
        self.model = model_instance.__class__
        self.options_dict = options_to_kwargs(self.model)
        self.p = self.model.parameter.flatten()

        self.state = self.model.state.flatten()
        self.dot = self.model.dot.flatten()

        self.initial_residual = self.model.initial_residual.flatten()
        self.residual = self.model.residual.flatten()
        self.output = self.model.output.flatten()

        self.final_time = self.model.tf
        self.start_time = self.model.t0

        self.state_count = self.state.shape[0]
        self.dot_count = self.dot.shape[0]
        self.total_count = self.state_count + self.dot_count
        self.count_diff = self.state_count - self.dot_count
        breakpoint()

        # solve for the initial conditions from the initial residuals
        class DAEInitialConditionSolve(AlgebraicSystem):
            residual_dict = ElementMap()
            for elem in self.model.parameter:
                residual_dict[elem] = parameter(
                    name=elem.name,
                    shape=elem.shape,
                )
            for elem in self.model.state:
                residual_dict[elem] = variable(
                    name=elem.name,
                    shape=elem.shape,
                    initializer=elem.initializer,
                )
            for elem in self.model.dot:
                backend_name = f"{elem.name}_dot"
                residual_dict[elem] = variable(
                    name=backend_name,
                    shape=elem.shape,
                    # initializer=elem.initializer, <- does this need an initializer?
                )
            input_dict = residual_dict.as_("backend_repr")
            for index, elem in enumerate(self.model.initial_residual):
                # breakpoint()
                residual(
                    substitute(elem.backend_repr, input_dict),
                    name=f"residual_{index}",
                )

        self.initial_conditions = DAEInitialConditionSolve(**model_instance.parameter)

        # self.symbolic_initial_condition = DAEInitialConditionSolve(
        #     **self.model.parameter
        # )
        # self._initial_condition_jacobian = jacobian(
        #     self.symbolic_initial_condition.variable.flatten(), self.p
        # )
        # self.initial_condition_jacobian = substitute(
        #     self._initial_condition_jacobian,
        #     {self.state[0]: 1, self.state[1]: 0, self.state[2]: 0},
        # )
        # self.sensitivity_initial_conditions_func = expression_to_operator(
        #     [self.p],
        #     self.initial_condition_jacobian,
        #     f"{self.model.__name__}_sensitivity_initial_condition",
        # )

        self.sens_state_initial_condition_func = expression_to_operator(
            [self.p], jacobian(self.state, self.p)
        )
        self.sens_dot_initial_condition_func = expression_to_operator(
            [self.state], jacobian(self.residual, self.p)
        )
        # for robertson it would be a 6x3
        # it would be symbolic, could use expression to op or substitute

        self.state_sensitivity = MX.sym(
            "s", self.p.shape[0], self.state_count
        )  # same shape as parameters * state (robertson would be 3x3)
        # casadi MX symbols
        self.dot_sensitivity = MX.sym("sd", self.p.shape[0], self.dot_count)
        self.sensitivity_residual_state = (
            jacobian(self.residual, self.state) @ self.state_sensitivity
        )
        self.sensitivity_residual_dot = (
            jacobian(self.residual, self.dot) @ self.dot_sensitivity
        )
        self.sensitivity_residual_parameter = jacobian(self.residual, self.p)

        self.sensitivity_residual = (
            self.sensitivity_residual_state
            + self.sensitivity_residual_dot
            + self.sensitivity_residual_parameter
        )

        self.sensitivity_vars = [
            self.state_sensitivity,
            self.dot_sensitivity,
            self.state,
            self.dot,
            self.p,
        ]
        self.sens_func = expression_to_operator(
            self.sensitivity_vars,
            self.sensitivity_residual,
            f"{self.model.__name__}_sensitivity_residual",
        )

        self.residual_vars = [
            self.state,
            self.dot,
            self.p,
        ]
        self.residual_func = expression_to_operator(
            self.residual_vars,
            self.residual,
            f"{self.model.__name__}_residual",
        )

        self.e_exprs = []  # this will be function (where the zeros should be found)
        self.h_exprs = []  # this will be update residual
        if isinstance(self.model.t0, BaseElement):
            t0 = self.model.t0.backend_repr
        elif isinstance(self.model.t0, (symbol_class, int, float, np.ndarray)):
            t0 = self.model.t0
        else:
            unexpcted_t0 = "unexpected value for t0"
            raise ValueError(unexpcted_t0)
        at_time_slices = [
            NextTimeFromSlice(
                expression_to_operator(
                    [self.p],
                    # TODO in future allow t0 to occur at arbitrary times
                    concat([t0, t0, inf]),
                    f"{self.model.__name__}_at_times_t0",
                )
            )
        ]

        terminating = []
        self.dae_model = dae_model = model_instance.__class__
        self.dae_events = dae_events = [e for e in dae_model.DAEEvent._meta.subclasses]
        if (
            not isinstance(self.model.tf, (np.ndarray, float))
            or not np.isinf(self.model.tf).any()
        ):

            class DAETerminate(dae_model.DAEEvent):
                at_time = (self.model.tf,)
                terminate = True

            dae_events += [DAETerminate]
            dae_model.DAEEvent._meta.subclasses = dae_model.DAEEvent._meta.subclasses[
                -1
            ]

        num_events = len(dae_events)
        for event_idx, event in enumerate(dae_events):
            if isnan(event.function) == isnan(event.at_time):
                msg = f"Event class `{event}` has set both `function` and `at_time`"
                raise ValueError(msg)
            if not isnan(getattr(event, "function", np.nan)):  # function event
                e_expr = event.function
            else:  # if the event if a time event
                at_time = event.at_time
                if hasattr(at_time, "__len__"):
                    if len(at_time) in [2, 3]:
                        at_time = slice(*tuple(at_time))
                    else:
                        at_time = at_time[0]

                if isinstance(at_time, slice):
                    if at_time.step is None:
                        raise ValueError

                    at_time_start = 0 if at_time.start is None else at_time.start

                    e_expr = (
                        at_time.step
                        * sin(pi * (self.model.t - at_time_start) / at_time.step)
                        / (pi * 100)
                    )
                    # self.events(solver_res.values.t, solver_res.values.y, gs)
                    e_expr = mod(self.model.t - at_time_start, at_time.step)

                    # TODO: verify start and stop for at_time slice
                    if isinstance(at_time_start, symbol_class) or at_time_start != 0.0:
                        e_expr = e_expr * (self.model.t >= at_time_start)
                        # if there is a start offset, add a linear term to provide a
                        # zero-crossing at first occurance
                        pre_term = (at_time_start - self.model.t) * (
                            self.model.t <= at_time_start
                        )
                    else:
                        pre_term = 0

                    if at_time.stop is not None:
                        e_expr = e_expr * (self.model.t <= at_time.stop)
                        # if there is an end-time, hold constant to prevent additional
                        # zero crossings -- hopefully works even if stop is on an event
                        # post_term = (
                        #     (ode_model.t >= at_time.stop)
                        #     * at_time.step
                        #     * casadi.sin(
                        #         casadi.pi
                        #         * (at_time.stop - at_time_start)
                        #         / at_time.step
                        #     )
                        #     / casasadi.pi
                        # )
                        post_term = (self.model.t >= at_time.stop) * mod(
                            at_time.stop - at_time_start, at_time.step
                        )
                        at_time_stop = at_time.stop
                    else:
                        post_term = 0
                        at_time_stop = inf

                    e_expr = e_expr + pre_term + post_term

                    at_time_slices.append(
                        NextTimeFromSlice(
                            expression_to_operator(
                                [self.p],
                                concat([at_time_start, at_time_stop, at_time.step]),
                                f"{ode_model.__name__}_at_times_{event_idx}",
                            )
                        )
                    )
                else:
                    if isinstance(at_time, BaseElement):
                        at_time0 = at_time.backend_repr
                    else:
                        at_time0 = at_time
                    e_expr = at_time0 - self.model.t
                    at_time_slices.append(
                        NextTimeFromSlice(
                            expression_to_operator(
                                [self.p],
                                concat([at_time0, at_time0, inf]),
                                f"{dae_model.__name__}_at_times_{event_idx}",
                            )
                        )
                    )

            self.e_exprs.append(e_expr)

            if event.terminate:
                terminating.append(event_idx)

            # h_expr = expression_to_operator(
            #     self.h_expr_vars,
            #     self.model.DAEEvent._meta.subclasses[0].update_residual.backend_repr,
            # )
            # self.h_exprs.append(h_expr)
        self.events = expression_to_operator(
            [self.state, self.dot, self.p, self.model.t],
            concat(self.e_exprs),
            f"{self.dae_model.__name__}_event",
        )
        breakpoint()

        if isinstance(self.model.tf, BaseElement):
            tf = self.model.tf.backend_repr
        elif isinstance(self.model.tf, (symbol_class, int, float, np.ndarray)):
            tf = self.model.tf
        else:
            unexpcted_tf = "unexpected value for tf"
            raise ValueError(unexpcted_tf)
        at_time_slices.append(
            NextTimeFromSlice(
                expression_to_operator(
                    [self.p],
                    # TODO in future allow t0 to occur at arbitrary times
                    concat([tf, tf, inf]),
                    f"{self.model.__name__}_at_times_tf",
                )
            )
        )

        self.model.DAEEvent._meta.subclasses[0]
        self.m

        breakpoint()
        self.dae_analysis_soln = DAESystemAnalysis(
            initial_conditions=self.initial_conditions,
            p=self.initial_conditions.parameter.flatten(),
            final_time=self.final_time,
            start_time=self.start_time,
            state_count=self.state_count,
            count_diff=self.count_diff,
            residual_func=self.residual_func,
            events=self.events,
            updates=self.h_exprs,
            terminating=terminating,
            sens_func=self.sens_func,
            sens_state_initial_condition_func=self.sens_state_initial_condition_func,
            sens_dot_initial_condition_func=self.sens_dot_initial_condition_func,
            time_generator=TimeGeneratorFromSlices(at_time_slices),
            **self.options_dict,
        )

        self(model_instance)

    def __call__(self, model_instance):
        soln = self.dae_analysis_soln()

        state_soln = np.stack([soln.y[:, x] for x in range(self.dot_count)])
        dot_soln = np.stack([soln.yp[:, x] for x in range(self.dot_count)])

        model_instance.t = soln.t
        model_instance.bind_field(model_instance.__class__.state.wrap(state_soln))
        model_instance.bind_field(model_instance.__class__.dot.wrap(dot_soln))
