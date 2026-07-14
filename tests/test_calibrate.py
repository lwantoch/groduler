import numpy as np
import pandas as pd
import pytest
from pytest import approx

from groduler.calibrate import (
    calibrate_footprint,
    calibrate_throughput,
    load_sweep,
    organize_runs,
)


@pytest.fixture
def medians() -> pd.DataFrame:
    """Small helper for generating testdata."""

    atom_size = np.array([2000.0, 5000.0, 20000.0, 120000.0])
    throughput = 500000.0 * atom_size ** (-0.75)  # A=500000, b=-0.75
    sm_active = 77.0 * atom_size / (atom_size + 8697.0)  # Fmax=77, K=8697

    medians = pd.DataFrame(
        {
            "gpu": ["L40S", "L40S", "L40S", "L40S", "RTX"],
            "atom_size": list(atom_size) + [2000],
            "colocation_depth": [1, 1, 1, 1, 1],
            "throughput": list(throughput) + [999.0],  # RTX = Müll
            "sm_active": list(sm_active) + [10.0],  # RTX = Müll
        }
    )

    return medians


def test_organize_runs():
    """ "Prooving function for organizing data"""

    test_data = pd.DataFrame(
        {
            "gpu": ["L40S", "L40S", "L40S"],
            "atom_size": [2000, 2000, 2000],
            "througput": [500, 450, 480],
            "sm_active": [97, 95, 96],
            "colocation_depth": [4, 4, 4],
        }
    )

    result = organize_runs(test_data)

    assert result.loc[0, "througput"] == 480
    assert result.loc[0, "sm_active"] == 96


def test_load_sweep(tmp_path):
    """Checking funtion for data loading"""

    sweep_data = tmp_path / "sweep.csv"
    sweep_data.write_text("gpu,atom_size\nL40S,2000\n")
    sweep_data = load_sweep(tmp_path / "sweep.csv")

    assert sweep_data.loc[0, "gpu"] == "L40S"
    assert sweep_data.loc[0, "atom_size"] == 2000


def test_calibrate_throughput(medians: pd.DataFrame):
    """Checking the calibration function for througput"""

    result = calibrate_throughput(medians, "L40S")

    assert result == calibrate_throughput(medians, "L40S")
    assert result.factor_A == approx(500000.0, rel=1e-3)
    assert result.factor_colocation == approx(-0.75, abs=1e-3)


def test_calibrate_footrint(medians: pd.DataFrame):
    """Checking the calibration function for footprint"""

    result = calibrate_footprint(medians, "L40S")

    assert result == calibrate_footprint(medians, "L40S")
    assert result.fmax == approx(77, rel=1e-3)
