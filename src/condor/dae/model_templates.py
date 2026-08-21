from condor.models import ModelTemplate, ModelType, SubmodelTemplate, SubmodelType
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
    asdict,
)
import numpy as np
from condor.dae.dae_implementation import DAEAnalysisImplementation
from condor.backend import process_relational_element, get_symbol_data


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
    t0 = placeholder(default=None)
    tf = placeholder(default=np.inf)

    parameter = FreeField()

    state = InitializedField(Direction.internal)
    dot = FreeMatchedField(state)

    initial_residual = FreeAssignedField(Direction.internal)
    residual = FreeAssignedField(Direction.internal)
    shared_residual = DuplicatingFreeAssignedField(
        Direction.internal, field_list=[residual, initial_residual]
    )

    output = AssignedField(Direction.output)


# copied from class Event in contrib.py
class DAEEventType(SubmodelType):
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
class DAEEvent(SubmodelTemplate, model_metaclass=DAEEventType, primary=DAESystem):
    update_residual = MatchedField(
        DAESystem.initial_residual,
        direction=Direction.output,
        default_factory=pass_through,
    )
    terminate = placeholder(default=False)
    function = placeholder(default=np.nan)
    at_time = placeholder(default=np.nan)
