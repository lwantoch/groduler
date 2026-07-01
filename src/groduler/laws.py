from typing import NamedTuple

import numpy as np
from scipy.optimize import curve_fit


class FootprintFit(NamedTuple):
    """Fitted constants of the Footpint law.

    Footprint law is the defined as percent of SMactive used by a job on one GPU and is depending of atomsize of the system

       sm_active=(fmax * atom_size)/(factor_k+atom_size)
    """

    fmax: float
    factor_k: float


def _footprint_law_model(atom_size: np.array, fmax: float, factor_k: float):
    """ "defining the model for optimization.

    Function definition for the fit."""

    smact = (fmax * atom_size) / (factor_k + atom_size)

    return smact


def fitting_law(atom_size: np.array, smact: np.array) -> tuple[float, float]:
    """Finding the footprint parameters by fitting with SciPy."""

    popt, pcov = curve_fit(_footprint_law_model, atom_size, smact, p0=[50, 500])

    fmax, factor_k = popt

    return fmax, factor_k
