from condor.models import (
    ModelTemplate,
    ModelType,
)
from condor.fields import (
    AssignedField,
    Direction,
    FreeAssignedField,
    FreeField,
    FreeMatchedField,
    InitializedField,
)
import numpy as np
from condor.dae.dae_implementation import DAEAnalysisImplementation
from condor.backend import process_relational_element


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

    # differential_state = FreeField(Direction.internal)
    differential_state = InitializedField(Direction.internal)
    # algebraic_state = FreeField(Direction.internal)
    algebraic_state = InitializedField(Direction.internal)
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
