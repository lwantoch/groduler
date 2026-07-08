from typing import NamedTuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress


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


def fitting_footprint_law(atom_size: np.array, smact: np.array) -> FootprintFit:
    """Finding the footprint parameters by fitting with SciPy."""

    popt, pcov = curve_fit(_footprint_law_model, atom_size, smact, p0=[50, 500])

    fmax, factor_k = popt

    return FootprintFit(fmax=fmax, factor_k=factor_k)


class ThroughputFit(NamedTuple):
    """Fitted constants of the Throughput law.

    Throughput law is the defined as percent of ns/day which one job can achieve

       throughput=factor_A*atom_size**factor_colocate
    """

    factor_A: float
    factor_colocation: float
    check_r2: float


def fitting_throughput_law(atom_size: np.array, throughput: np.array) -> ThroughputFit:
    """Finding the throughput parameters by fitting with SciPy."""

    fit_throughput = linregress(np.log(atom_size), np.log(throughput))

    factor_colocation = fit_throughput.slope
    factor_A = np.exp(fit_throughput.intercept)
    check_r2 = fit_throughput.rvalue

    return ThroughputFit(
        factor_A=factor_A, factor_colocation=factor_colocation, check_r2=check_r2
    )


def optimal_packing(packing_depths: np.ndarray, system_throughput: np.ndarray) -> int:
    """It finds the best packing composition of system of given class"""

    best_packing = packing_depths[np.argmax(system_throughput)]

    return best_packing
