import numpy as np
from pytest import approx

from groduler.laws import fitting_law


def test_fitting_law_recovers_known_constants():

    atom_size = np.array([2000.0, 5000.0, 20000.0, 120000.0])
    smact = np.array([15.0, 28.0, 54.0, 72.0])

    fmax, factor_k = fitting_law(atom_size, smact)

    assert fmax == approx(77.0, abs=1.0)
