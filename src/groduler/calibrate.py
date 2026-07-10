import pandas as pd


def organize_runs(sweep: pd.DataFrame) -> pd.DataFrame:
    # reading and formatting the simulation data.

    return (
        sweep.groupby(["gpu", "atom_size", "colocation_depth"]).median().reset_index()
    )
