import pandas as pd

from groduler.calibrate import organize_runs

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
    """ "Proving function for organizing data"""

    result = organize_runs(test_data)

    assert result.loc[0, "througput"] == 480
    assert result.loc[0, "sm_active"] == 96
