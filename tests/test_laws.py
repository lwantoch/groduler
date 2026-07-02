import numpy as np
from pytest import approx

from groduler.laws import fitting_footprint_law, fitting_throughput_law, optimal_packing


def test_fitting_footprint_law_recovers_known_constants():

    atom_size = np.array([2000.0, 5000.0, 20000.0, 120000.0])
    smact = np.array([15.0, 28.0, 54.0, 72.0])

    fmax, factor_k = fitting_footprint_law(atom_size, smact)

    assert fmax == approx(77.0, abs=1.0)


def test_fitting_throughput_law_recovers_known_constants():

    atom_size = np.array([2000.0, 5000.0, 20000.0, 120000.0])
    throughput = 5000 * (atom_size**20)

    factor_colocation, factor_A, check_r2 = fitting_throughput_law(
        atom_size, throughput
    )

    assert factor_A == approx(5000.0, abs=1.0)
    assert factor_colocation == approx(20.0, abs=1.0)


def test_optimal_packing():

    system_throughput = [10, 200, 5657, 2836, 56, 909]
    packing_depths = [2, 5, 6, 8, 12, 16]
    best_packing = packing_depths[np.argmax(system_throughput)]
    assert best_packing == 6
