import numpy as np


def split_systems(atom_size: np.ndarray) -> np.ndarray:
    """Splitting systems into 3 categories small, medium and large.


    The splitting is just arbitrary, the Tresholds may be adapted."""

    _THRESHOLDS = [20_000, 120_000]
    _LABELS = np.array(["small", "medium", "large"])
    return _LABELS[np.digitize(atom_size, _THRESHOLDS)]
