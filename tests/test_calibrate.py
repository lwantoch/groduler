import pandas as pd

from groduler.calibrate import load_sweep, organize_runs

test_data = pd.DataFrame(
    {
        "gpu": ["L40S", "L40S", "L40S"],
        "atom_size": [2000, 2000, 2000],
        "througput": [500, 450, 480],
        "sm_active": [97, 95, 96],
        "colocation_depth": [4, 4, 4],
    }
)


def test_organize_runs():
    """ "Prooving function for organizing data"""

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
