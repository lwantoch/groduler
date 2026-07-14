import numpy as np

from groduler.split import split_systems


def test_splitting():
    atom_size = np.array([2_000.0, 5_000.0, 20_000.0, 120_000, 150_000])
    classes = split_systems(atom_size)
    expected = ["small", "small", "medium", "large", "large"]

    np.testing.assert_array_equal(classes, expected)
