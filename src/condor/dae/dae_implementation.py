from condor.backend.operators import substitute, concat, inf
from condor.utils import ElementMap

from condor.backend import expression_to_operator, symbol_class
from condor import AlgebraicSystem
from condor.dae.solvers import (
    DAESystemAnalysis,
    TimeGeneratorFromSlices,
    NextTimeFromSlice,
)
from condor.implementations.utils import options_to_kwargs
import numpy as np
from condor.fields import BaseElement


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
        breakpoint()
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

        self.dae_analysis_soln = DAESystemAnalysis(
            initial_conditions=self.initial_conditions,
            p=self.initial_conditions.parameter.flatten(),
            final_time=self.final_time,
            start_time=self.start_time,
            state_count=self.state_count,
            count_diff=self.count_diff,
            residual_func=self.residual_func,
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
