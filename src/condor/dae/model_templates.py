from condor.models import (
    ModelTemplate,
    ModelType,
    SubmodelTemplate,
    SubmodelType,
    SubmodelMetaData,
)
from condor.fields import (
    Field,
    AssignedField,
    Direction,
    FreeAssignedField,
    FreeField,
    FreeElement,
    FreeMatchedField,
    InitializedField,
    MatchedField,
    pass_through,
    WithDefaultField,
    TrajectoryOutputField,
    asdict,
    zero_like,
)
import numpy as np
from condor.dae.dae_implementation import DAEAnalysisImplementation
from condor.contrib import TrajectoryAnalysisMetaData, TrajectoryAnalysisType
from condor.backend import process_relational_element, get_symbol_data
from dataclasses import dataclass, field


class DuplicatingFreeAssignedField(FreeAssignedField, element_class=FreeElement):
    def __init__(self, direction=None, field_list=None, **kwargs):
        super().__init__(direction=direction, **kwargs)
        self._field_list = field_list
        self._init_kwargs.update(field_list=field_list)

    def __call__(self, value, **kwargs):
        for field in self._field_list:
            field(value)

        return super().__call__(value, **kwargs)


# model type
class DAESystemType(ModelType):
    @classmethod
    def process_placeholders(cls, new_cls, attrs):
        super().process_placeholders(new_cls, attrs)
        for elem in new_cls.residual:
            process_relational_element(elem)
        for elem in new_cls.initial_residual:
            process_relational_element(elem)

    implementation = DAEAnalysisImplementation


# model template
class DAESystem(ModelTemplate, model_metaclass=DAESystemType):
    t = placeholder(default=None)

    parameter = FreeField()

    state = InitializedField(Direction.internal)
    dot = FreeMatchedField(state)

    initial_residual = FreeAssignedField(Direction.internal)
    residual = FreeAssignedField(Direction.internal)
    shared_residual = DuplicatingFreeAssignedField(
        Direction.internal,
        field_list=[residual, initial_residual],
    )
    modal = WithDefaultField(Direction.internal)
    output = AssignedField(Direction.output)


# copied from class Event in contrib.py
# @dataclass
class DAEAnalysisMetaData(SubmodelMetaData):
    events: list = field(default_factory=list)
    modes: list = field(default_factory=list)


class DAEAnalysisType(SubmodelType):
    metadata_class = DAEAnalysisMetaData  # DAEEventMetaData

    @classmethod
    def __prepare__(
        cls,
        *args,
        include_events=None,
        exclude_events=None,
        include_modes=None,
        exclude_modes=None,
        **kwargs,
    ):
        cls_dict = super().__prepare__(*args, **kwargs)
        if exclude_events is not None and include_events is not None:
            msg = "Use only one of include or exclude events"
            raise ValueError(msg)
        if include_events is None:
            cls_dict.meta.events = list(cls_dict.meta.primary.Event)
        else:
            cls_dict.meta.events = include_events

        if exclude_events is not None:
            cls_dict.meta.events = [
                event for event in cls_dict.meta.events if event not in exclude_events
            ]

        if exclude_modes is not None and include_modes is not None:
            msg = "Use only one of include or exclude modes"
            raise ValueError(msg)

        if include_modes is None:
            cls_dict.meta.modes = list(cls_dict.meta.primary.Mode)
        else:
            cls_dict.meta.modes = include_modes

        if exclude_modes is not None:
            cls_dict.meta.modes = [
                mode for mode in cls_dict.meta.modes if mode not in exclude_modes
            ]
        return cls_dict

    def __new__(
        cls,
        *args,
        include_events=None,
        exclude_events=None,
        include_modes=None,
        exclude_modes=None,
        **kwargs,
    ):
        new_cls = super().__new__(cls, *args, **kwargs)
        return new_cls

    # @classmethod
    # def bind_model_fields(cls, new_cls, attrs):
    #     super().bind_model_fields(new_cls, attrs)
    #     breakpoint()
    #     p = new_cls.parameter.flatten()
    #     x = new_cls.state.flatten()

    #     ode_model = new_cls._meta.primary
    #     model = new_cls
    #     control_subs_pairs = {
    #         control.backend_repr: [control.default] for control in ode_model.modal
    #     }
    #     for mode in model._meta.modes:
    #         for act in mode.action:
    #             control_subs_pairs[act.match.backend_repr].insert(
    #                 -1, (mode.condition, act.backend_repr)
    #             )
    #     control_sub_expression = {}
    #     for k, v in control_subs_pairs.items():
    #         control_sub_expression[k] = substitute(if_else(*v), control_sub_expression)

    #     new_cls._meta.initial_condition_function = expression_to_operator(
    #         [p],
    #         substitute(new_cls.initial.flatten(), control_sub_expression),
    #         f"{new_cls.__name__}_initial_condition",
    #     )

    #     new_cls._meta.state_equation_function = expression_to_operator(
    #         [new_cls.t, x, p],
    #         substitute(new_cls.dot.flatten(), control_sub_expression),
    #         f"{new_cls.__name__}_state_equation",
    #     )
    #     new_cls._meta.output_equation_function = expression_to_operator(
    #         [new_cls.t, x, p],
    #         substitute(new_cls.dynamic_output.flatten(), control_sub_expression),
    #         f"{new_cls.__name__}_output_equation",
    #     )
    #     new_cls._meta.modal_eval = expression_to_operator(
    #         [new_cls.t, x, p],
    #         substitute(new_cls.modal.flatten(), control_sub_expression),
    #         f"{new_cls.__name__}_modal_evaluation",
    #     )
    implementation = DAEAnalysisImplementation


class DAEAnalysis(
    SubmodelTemplate,
    model_metaclass=DAEAnalysisType,  # DAEAnalysisType,
    primary=DAESystem,
    copy_fields=True,
    copy_embedded_models=False,
):
    trajectory_output = TrajectoryOutputField()
    t0 = placeholder(default=None)
    tf = placeholder(default=np.inf)

    # class Options:
    #     __implementation__ = DAEAnalysisImplementation


class EventType(SubmodelType):
    @classmethod
    def process_condor_attr(cls, attr_name, attr_val, new_cls):
        state_elem = new_cls._meta.primary.state.get(backend_repr=attr_val)
        if state_elem != []:
            primary_attr = getattr(new_cls._meta.primary, state_elem.name, None)
            if primary_attr is None:
                check_attr_name(state_elem.name, attr_val, new_cls._meta.primary)
                setattr(new_cls._meta.primary, state_elem.name, state_elem)
                new_cls._meta.primary._meta.user_set[state_elem.name] = attr_val
            elif primary_attr is not state_elem:
                msg = (
                    f"{new_cls} attempting to assign state {attr_name} = {attr_val}"
                    f" but {new_cls._meta.primary} already has {attr_name} ="
                    f"{primary_attr}"
                )
                raise NameError(msg)
            if state_elem.name != attr_name:
                super().process_condor_attr(attr_name, attr_val, new_cls)
        else:
            super().process_condor_attr(attr_name, attr_val, new_cls)


# sub model templates for events (similar to class Event())
# instead of update something like reinitialize residual
# think about API for shared residual
class Event(
    SubmodelTemplate,
    model_metaclass=EventType,
    primary=DAESystem,
):
    update_residual = FreeAssignedField(Direction.internal)
    update = FreeMatchedField(
        DAESystem.state,
        direction=Direction.internal,
        default_factory=pass_through,
    )
    update_dot = FreeMatchedField(
        DAESystem.state,
        direction=Direction.internal,
        default_factory=pass_through,
    )
    terminate = placeholder(default=False)
    function = placeholder(default=np.nan)
    at_time = placeholder(default=np.nan)

    # shared_update_residual = DuplicatingFreeAssignedField(
    #     Direction.internal, field_list=[update_residual, DAESystem.residual]
    # )


# TODO: make something like TrajectoryAnalysis = DAEAnalysis
# TODO: make Mode sub model template

# update_residual = FreeMatchedField(
#     DAESystem.initial_residual,
#     direction=Direction.output,
#     default_factory=pass_through,
# )


class Mode(SubmodelTemplate, primary=DAESystem):
    condition = placeholder(default=1.0)
    action = MatchedField(
        DAESystem.modal, direction=Direction.internal, default_factory=zero_like
    )


def copy_field(new_model_name, old_field, new_field=None):
    if new_field is None:
        new_field = old_field.inherit(new_model_name, field_type_name=old_field._name)
    new_field._elements = [sym for sym in old_field]
    return new_field


# DAESystem.shared_residual.field_list.append(DAEEvent.update_residual)
