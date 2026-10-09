from condor.backend.operators import substitute, concat, inf, jacobian
from condor.utils import ElementMap

from condor.backend import expression_to_operator, symbol_class, FunctionOperator
from condor import AlgebraicSystem
from condor.dae.solvers_adjoint import DAESystem, TrajectoryAnalysis
from condor.implementations.utils import options_to_kwargs
import condor.solvers.sweeping_gradient_method as sgm
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
        self.dae_model = dae_model = self.model._meta.primary
        self.options_dict = options_to_kwargs(self.model)
        self.p = self.model.parameter.flatten()

        self.state = self.model.state.flatten()
        self.dot = self.model.dot.flatten(on_field=self.model.state)
        # self.dot = self.model.dot.flatten()

        self.initial_residual = self.model.initial_residual.flatten()
        self.residual = self.model.residual.flatten()
        self.output = self.model.output.flatten()

        self.final_time = self.model.tf
        self.start_time = self.model.t0

        self.total_count = self.model.state._count + self.model.dot._count
        self.count_diff = self.model.state._count - self.model.dot._count

        self.simulation_signature = [
            self.model.t,
            self.state,
            self.dot,
            self.p,
        ]
        breakpoint()
        self.residual_func = expression_to_operator(
            self.simulation_signature,
            self.residual,
            f"{self.model.__name__}_residual",
        )

        self.trajectory_output_expr = self.model.trajectory_output.flatten()
        self.can_sgm = isinstance(self.p, symbol_class) and isinstance(
            self.trajectory_output_expr, symbol_class
        )

        self.trajectory_output_integrand = self.model.trajectory_output.flatten(
            "integrand"
        )
        self.trajectory_output_integrand_func = expression_to_operator(
            self.simulation_signature,
            self.trajectory_output_integrand,
            f"{self.model.__name__}_trajectory_output_integrand",
        )

        self.trajectory_output_terminal_term = self.model.trajectory_output.flatten(
            "terminal_term"
        )
        self.trajectory_output_terminal_term_func = expression_to_operator(
            self.simulation_signature,
            self.trajectory_output_terminal_term,
            f"{self.model.__name__}_trajectory_output_terminal_term",
        )

        # solve for the initial conditions from the initial residuals
        class DAEInitialConditionSolve(AlgebraicSystem):
            dae_model = self.model._meta.primary
            breakpoint()
            input_dict = ElementMap()

            p_var = parameter(shape=dae_model.parameter._count)
            p_wrap = dae_model.parameter.wrap(p_var)
            for idx, elem in enumerate(dae_model.parameter):
                input_dict[elem.backend_repr] = p_wrap[elem.name]

            state_var = variable(
                shape=dae_model.state._count,
                initializer=dae_model.state.flatten("initializer").reshape(-1, 1),
            )  # , initializer=1)
            state_wrap = dae_model.state.wrap(state_var)
            breakpoint()
            for idx, elem in enumerate(dae_model.state):
                # breakpoint()
                input_dict[elem.backend_repr] = state_wrap[elem.name]
            # for idx, elem in enumerate(self.model.state):
            #     input_dict[elem.backend_repr] = state_wrap.flatten()[idx]

            dot_var = variable(shape=dae_model.dot._count)
            dot_wrap = dae_model.state.wrap(dot_var)
            for idx, elem in enumerate(dae_model.dot):
                # breakpoint()
                input_dict[elem.backend_repr] = dot_wrap[elem.name]
            # breakpoint()
            for index, elem in enumerate(dae_model.initial_residual):
                # breakpoint()
                residual(
                    substitute(elem.backend_repr, input_dict),
                    name=f"residual_{index}",
                )

        breakpoint()
        # check_state_dict = {
        #     DAEInitialConditionSolve.parameter.p_var: self.p,
        #     DAEInitialConditionSolve.variable.state_var: self.state,
        #     DAEInitialConditionSolve.variable.dot_var: self.dot,
        # }
        # check_op = expression_to_operator(
        #     [self.p, self.state, self.dot],
        #     substitute(
        #         DAEInitialConditionSolve.residual.flatten(),
        #         check_state_dict,
        #     ),
        # )
        # print(check_op([1, 0.1], [1, 0.1, 0], [0, 0, 0]))
        # breakpoint()
        # J = jacobian(
        #     DAEInitialConditionSolve.residual.flatten(),
        #     DAEInitialConditionSolve.variable.flatten(),
        # )
        # test_dict = {DAEInitialConditionSolve.parameter.p_var: self.p}
        # test_op = expression_to_operator(
        #     self.simulation_signature, substitute(J, test_dict)
        # )
        # rank_J = np.linalg.matrix_rank(
        #     test_op(0, [1, 0.1, -1.01], [0.1, -1.01], [1, 0.1]).full()
        # )
        # print(rank_J)
        # breakpoint()
        DAEInitialConditionSolve(self.p)
        breakpoint()

        def ic_solver(p):
            breakpoint()
            ic_solved = DAEInitialConditionSolve(p)
            return ic_solved.state_var, ic_solved.dot_var
            """
                within IC, create 1 param = size of DAE param
                2 var = x, xdot
                to get sub for residual, can use DAE model.state.wrap(x var) then substitute 
                construct sub dict from backend repr, state wrapped give names
                do same for parameter
                dot may do same thing, model.dot.wrap(xdot var) then do same thing
                do for all solvers (IC + event)
                possible to make function to create parameter and variables for me
            """

        # self.initial_conditions = DAEInitialConditionSolve(**model_instance.parameter)

        # function that takes the DAEICSolver, dummy function (wrapped solver),
        # call solver and split state and dot, function then gets passed to solver
        # solver gets called at beginning of trajectory analysis and
        # returns IC for states and dot separated
        # used for both ICs and updates

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

        # self.sens_state_initial_condition_func = expression_to_operator(
        #     [self.p], jacobian(self.state, self.p)
        # )
        # self.sens_dot_initial_condition_func = expression_to_operator(
        #     [self.state], jacobian(self.residual, self.p)
        # )
        # # for robertson it would be a 6x3
        # # it would be symbolic, could use expression to op or substitute

        # self.state_sensitivity = MX.sym(
        #     "s", self.p.shape[0], self.state_count
        # )  # same shape as parameters * state (robertson would be 3x3)
        # # casadi MX symbols
        # self.dot_sensitivity = MX.sym("sd", self.p.shape[0], self.dot_count)
        # self.sensitivity_residual_state = (
        #     jacobian(self.residual, self.state) @ self.state_sensitivity
        # )
        # self.sensitivity_residual_dot = (
        #     jacobian(self.residual, self.dot) @ self.dot_sensitivity
        # )
        # self.sensitivity_residual_parameter = jacobian(self.residual, self.p)

        # self.sensitivity_residual = (
        #     self.sensitivity_residual_state
        #     + self.sensitivity_residual_dot
        #     + self.sensitivity_residual_parameter
        # )

        # self.sensitivity_vars = [
        #     self.state_sensitivity,
        #     self.dot_sensitivity,
        #     self.state,
        #     self.dot,
        #     self.p,
        # ]
        # self.sens_func = expression_to_operator(
        #     self.sensitivity_vars,
        #     self.sensitivity_residual,
        #     f"{self.model.__name__}_sensitivity_residual",
        # )

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
            sgm.NextTimeFromSlice(
                expression_to_operator(
                    [self.p],
                    # TODO in future allow t0 to occur at arbitrary times
                    concat([t0, t0, inf]),
                    f"{self.model.__name__}_at_times_t0",
                )
            )
        ]

        terminating = []
        time_trigger = []
        self.dae_events = dae_events = [e for e in dae_model.Event._meta.subclasses]

        if (
            not isinstance(self.model.tf, (np.ndarray, float))
            or not np.isinf(self.model.tf).any()
        ):

            class DAETerminate(dae_model.Event):
                at_time = (self.model.tf,)
                terminate = True

            dae_events += [DAETerminate]  # this is internal for tf

            self.event_solvers = []
            for EventModel in self.dae_model.Event._meta.subclasses:
                if EventModel.update_residual._count > 0:

                    class DAEEventUpdateSolver(
                        AlgebraicSystem, name=f"{EventModel.__name__}UpdateSolver"
                    ):
                        dae_model = self.dae_model

                        if (
                            EventModel.update_residual._count
                            == self.model.shared_residual._count
                        ):
                            for idx, elem in enumerate(self.model.state):
                                EventModel.update_dot[elem.name]
                                EventModel.update_dot._elements[idx].name = elem.name

                        input_dict = ElementMap()
                        residual_swap_dict = ElementMap()

                        t_parameter = parameter(shape=1)
                        input_dict[dae_model.t] = t_parameter

                        p_backend = parameter(shape=dae_model.parameter._count)
                        p_wrap = dae_model.parameter.wrap(p_backend)
                        for idx, elem in enumerate(self.model.parameter):
                            input_dict[elem.backend_repr] = p_wrap.flatten()[idx]

                        p_state_backend = parameter(shape=dae_model.state._count)
                        p_state_wrap = dae_model.state.wrap(p_state_backend)
                        for idx, elem in enumerate(self.model.state):
                            input_dict[elem.backend_repr] = p_state_wrap.flatten()[idx]

                        update_state_backend = variable(shape=dae_model.state._count)
                        update_state_wrap = dae_model.state.wrap(update_state_backend)
                        for idx, elem in enumerate(EventModel.update):
                            input_dict[elem.backend_repr] = update_state_wrap.flatten()[
                                idx
                            ]
                            residual_swap_dict[
                                dae_model.state._elements[idx].backend_repr
                            ] = elem.backend_repr

                        p_dot_backend = parameter(shape=dae_model.dot._count)
                        p_dot_wrap = dae_model.dot.wrap(p_dot_backend)
                        for idx, elem in enumerate(self.model.dot):
                            input_dict[elem.backend_repr] = p_dot_wrap.flatten()[idx]

                        update_dot_backend = variable(shape=dae_model.dot._count)
                        update_dot_wrap = dae_model.dot.wrap(update_dot_backend)
                        for idx, elem in enumerate(EventModel.update_dot):
                            input_dict[elem.backend_repr] = update_dot_wrap.flatten()[
                                idx
                            ]
                            residual_swap_dict[
                                dae_model.dot._elements[idx].backend_repr
                            ] = elem.backend_repr

                        for index, elem in enumerate(EventModel.update_residual):
                            residual(
                                substitute(elem.backend_repr, input_dict),
                                name=f"residual_{index}",
                            )
                        next_idx = index + 1
                        for index, elem in enumerate(self.model.residual):
                            residual(
                                substitute(
                                    substitute(elem.backend_repr, residual_swap_dict),
                                    input_dict,
                                ),
                                name=f"residual_{index + next_idx}",
                            )

                    def wrapped_solver(t, p, y, yp):
                        solved = DAEEventUpdateSolver(t, p, y, yp)

                        return solved.update_state_backend, solved.update_dot_backend

                    self.event_solvers.append(wrapped_solver)
                else:

                    def pass_through_update(t, p, y, yp):
                        return y, yp

                    self.event_solvers.append(pass_through_update)

                    # for idx, elem in enumerate(self.model.state):
                    #     residual(
                    #         update_state_wrap.flatten()[idx]
                    #         == p_state_wrap.flatten()[idx],
                    #         name=f"residual_{idx}",
                    #     )
                    #     next_idx = idx + 1
                    # for idx, elem in enumerate(self.model.dot):
                    #     residual(
                    #         update_dot_wrap.flatten()[idx]
                    #         == p_dot_wrap.flatten()[idx],
                    #         name=f"residual_{idx + next_idx}",
                    #     )

                # if else, if -> append wrapped, else -> append pass_through
            dae_model.Event._meta.subclasses = dae_model.Event._meta.subclasses[:-1]

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
                        sgm.NextTimeFromSlice(
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
                        sgm.NextTimeFromSlice(
                            expression_to_operator(
                                [self.p],
                                concat([at_time0, at_time0, inf]),
                                f"{dae_model.__name__}_at_times_{event_idx}",
                            )
                        )
                    )

            self.e_exprs.append(e_expr)
            # breakpoint()
            if event.terminate:
                terminating.append(event_idx)
            if not isnan(event.at_time):
                time_trigger.append(event_idx)

            # if event.at_time:

        self.events_func = expression_to_operator(
            [self.state, self.dot, self.p, self.model.t],
            concat(self.e_exprs),
            f"{self.dae_model.__name__}_event",
        )

        if isinstance(self.model.tf, BaseElement):
            tf = self.model.tf.backend_repr
        elif isinstance(self.model.tf, (symbol_class, int, float, np.ndarray)):
            tf = self.model.tf
        else:
            unexpcted_tf = "unexpected value for tf"
            raise ValueError(unexpcted_tf)
        at_time_slices.append(
            sgm.NextTimeFromSlice(
                expression_to_operator(
                    [self.p],
                    # TODO in future allow t0 to occur at arbitrary times
                    concat([tf, tf, inf]),
                    f"{self.model.__name__}_at_times_tf",
                )
            )
        )

        breakpoint()
        self.dae_analysis_soln = DAESystem(
            dae_initial_conditions_solver=ic_solver,
            p=model_instance.parameter,
            final_time=self.final_time,
            start_time=self.start_time,
            state_count=self.model.state._count,
            count_diff=self.count_diff,
            output_count=self.model.trajectory_output._count,
            residual_func=self.residual_func,
            events=self.events_func,
            event_solvers=self.event_solvers,
            terminating=terminating,
            time_triggering=time_trigger,
            # sens_func=self.sens_func,
            # sens_state_initial_condition_func=self.sens_state_initial_condition_func,
            # sens_dot_initial_condition_func=self.sens_dot_initial_condition_func,
            integrand_terms=self.trajectory_output_integrand_func,
            terminal_terms=self.trajectory_output_terminal_term_func,
            time_generator=sgm.TimeGeneratorFromSlices(at_time_slices),
            **self.options_dict,
        )

        self.nominal_trajectory_analysis = TrajectoryAnalysis(
            state_system=self.dae_analysis_soln,
            integrand_terms=self.trajectory_output_integrand_func,
            terminal_terms=self.trajectory_output_terminal_term_func,
        )
        self.callback = FunctionOperator(
            function=self.nominal_trajectory_analysis,
            get_jacobian_func=None,  # self.generate_sgm_jacobian if self.can_sgm else None
            input_symbol=self.p,
            output_symbol=self.trajectory_output_expr,
            implementation=self,
            jacobian_of=None,
        )
        # update to CVODE branch
        self(model_instance)

    def generate_sgm_jacobian():
        pass

    def __call__(self, model_instance):
        self.soln = self.dae_analysis_soln(p=self.p)

        # self.callback.from_implementation = True
        # self.args = model_instance.parameter.flatten()
        # self.out = self.callback(self.args)
        self.out = 0
        # self.callback.from_implementation = False

        model_instance.t = self.soln.t

        dot_soln_list = []
        dot_wrap = model_instance.__class__.dot.wrap(self.soln.yp.T)
        for elem in model_instance.__class__.dot:
            dot_soln_list.append(dot_wrap[elem.name])
        dot_soln = np.concatenate(
            [item.reshape(-1, len(self.soln.yp)) for item in dot_soln_list]
        )
        # breakpoint()
        # test = fine[:2]
        # model_instance.bind_field(model_instance.__class__.dot.wrap(dot_soln))
        # model_instance.bind_field(model_instance.__class__.dot.wrap(self.soln.yp.T))
        breakpoint()
        model_instance.bind_field(model_instance.__class__.state.wrap(self.soln.y.T))
        model_instance.dot = model_instance.__class__.state.wrap(self.soln.yp.T)
        # model_instance.bind_field(
        #     model_instance.__class__.dot.wrap(self.soln.yp.T.reshape(3, 1, 2111))
        # )
        model_instance.bind_field(
            model_instance.__class__.trajectory_output.wrap(self.out)
        )

        # model_instance.bind_field(model_instance.__class__.output.wrap(soln.e))
