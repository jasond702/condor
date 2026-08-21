import matplotlib.pyplot as plt
import numpy as np

from condor import AlgebraicSystem
from condor.dae.model_templates import DAESystem
from condor.backend.operators import pi


class MultibodyCrankProblem(DAESystem):
    crank_length_m = parameter()  # a
    mass_crank_kg = parameter()  # m1
    MOI_crank_kg_m2 = parameter()  # J1

    rod_length_m = parameter()  # 2 meters
    mass_rod_kg = parameter()  # m2
    MOI_rod_kg_m2 = parameter()  # J2

    spring_const = parameter()  # k
    damp_coeff = parameter()  # c
    TSD_free_length = parameter()  # l
    const_hori_force = parameter()  # F

    y1 = state(initializer=pi/2)
    y2 = state(initializer=0)
    y3 = state(initializer=np.arcsin(-mass_crank_kg))
    v1 = state(initializer=0)
    v2 = state(initializer=0)
    v3 = state(initializer=0)

    # lambda and mu are algebraic states
    lam1 = state(initializer=0)
    lam2 = state(initializer=0)
    mu1 = state(initializer=0)
    mu2 = state(initializer=0)

    t0 = 0.0
    tf = 10.0

    initial_residual(y1 == pi / 3)
    initial_residual(y2 == np.sin(y3))
    initial_residual(y3 == np.arcsin(-mass_crank_kg))
    initial_residual(v1 == 0)
    initial_residual(v2 == 0)
    initial_residual(v3 == 0)
    initial_residual(lam1 == 0)
    initial_residual(lam2 == 0)
    initial_residual(mu1 == 0)
    initial_residual(mu2 == 0)
    initial_residual(dot[y1] == v1)
    initial_residual(dot[y2] == v2)
    initial_residual(dot[y3] == v3)
    initial_residual(dot[v1]==)
    initial_residual(dot[v2]==)
    initial_residual(dot[v3]==)
    initial_residual(dot[lam1]==0)
    initial_residual(dot[lam2]==0)
    initial_residual(dot[mu1]==0)
    initial_residual(dot[mu2]==0)

    Q = [None] * 3
    l = (
        y2**2
        - y2 * (np.cos(y3) + mass_crank_kg * np.cos(y1))
        + ((1 + (mass_crank_kg**2)) / 4)
        + (mass_crank_kg / 2) * np.cos(y3 - y1)
    ) ** 0.5
    ld = (1 / (2 * l)) * (
        2 * y2 * v2
        - v2 * (np.cos(y3) + mass_crank_kg * np.cos(y1))
        + y2 * (v3 * np.sin(y3) + mass_crank_kg * v1 * np.sin(y1))
        - (mass_crank_kg / 2) * (v3 - v1) * np.sin(y3 - y1)
    )
    f = spring_const * (l - TSD_free_length) + damp_coeff * ld

    Q[0] = (
        -(f / l) * (mass_crank_kg / 2) * ((1 / 2) * np.sin(y3 - y1) + y2 * np.sin(y1))
    )
    Q[1] = (f / l) * (
        (1 / 2) * np.cos(y3) - y2 + (mass_crank_kg / 2) * np.cos(y1)
    ) + const_hori_force
    Q[2] = -(f / l) * (1 / 2) * (
        y2 * np.sin(y3) - (mass_crank_kg / 2) * np.sin(y3 - y1)
    ) + const_hori_force * np.sin(y3)

    residual(
        dot[y1]
        == v1 - mass_crank_kg * mu1 * np.sin(y1) + mass_crank_kg * np.cos(y1) * mu2
    )
    residual(dot[y2] == v2 + mu1)
    residual(dot[y3] == v3 - mu1 * np.sin(y3) + mu2 * np.cos(y3))

    residual(
        MOI_crank_kg_m2 * dot[v1]
        == Q[0] - mass_crank_kg * lam1 * np.sin(y1) + mass_crank_kg * lam2 * np.cos(y1)
    )
    residual(mass_rod_kg * dot[v2] == Q[1] - lam1)
    residual(MOI_rod_kg_m2 * dot[v3] == Q[2] - lam1 * np.sin(y3) + lam2 * np.cos(y3))

    residual(y2 * np.cos(y3) - mass_crank_kg * np.cos(y1) == 0)
    residual(-np.sin(y3) - mass_crank_kg * np.sin(y1) == 0)

    residual(mass_crank_kg * v1 * np.sin(y1) + v2 * v3 * np.sin(y3) == 0)
    residual(-mass_crank_kg * v1 * np.cos(y1) - v2 * np.cos(y3) == 0)

    class Options:
        num_steps = 100


sim = MultibodyCrankProblem(
    crank_length_m=1 / 2,
    mass_crank_kg=1.0,
    MOI_crank_kg_m2=1,
    rod_length_m=2.0,
    mass_rod_kg=1.0,
    MOI_rod_kg_m2=2.0,
    spring_const=1.0,
    damp_coeff=1.0,
    TSD_free_length=1.0,
    const_hori_force=1.0,
)
