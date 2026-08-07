"""
▼ ◄ ▲ ► ▼ ◄ ▲ ► ▼ ◄ ▲ ► ▼ ◄ ▲ ► ▼ ◄ ▲ ▼ ◄ ▲ ► ▼ ◄ ▲ ► ▼ ◄▼
◄ ▲ ► ▼ ◄ ▲ ► ▼Sorry, I've dropped my bag of Doritos™ brand
chips▲ ► ▼ ◄ ▲ ► ▼ ◄ ▲ ▼ ◄ ▲ ► ▼ ◄ ▲ ► ▼ ◄ ▲ ► ▼ ◄ ▲ ► ▼ ►
▼ ◄ ◄ ▲▲ ► ▼ ◄▼ ◄ ◄ ▼ ◄ ▲ ► ▼ ◄ ▲ ► ▼ ◄ ▲ ► ▼ ◄ ▲ ►

Future example problem, a bouncing ball problem! :)
"""

import matplotlib.pyplot as plt
import numpy as np

from condor.dae.model_templates import DAESystem
from condor.backend.operators import pi


class BouncingBallProblem(DAESystem):
    mass_kg = parameter()
    g_const = parameter()

    lam = algebraic_state()
    height = differential_state()
    velocity = differential_state()

    residual()
    residual()
    residual()
