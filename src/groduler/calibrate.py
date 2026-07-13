import pandas as pd


def organize_runs(sweep: pd.DataFrame) -> pd.DataFrame:
    # reading and formatting the simulation data.

    return (
        sweep.groupby(["gpu", "atom_size", "colocation_depth"]).median().reset_index()
    )


def load_sweep(path: str) -> pd.DataFrame:
    # Function loading data for calibrations

    sweep_data = pd.read_csv(path)

    return sweep_data
