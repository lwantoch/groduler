import pandas as pd

from groduler.laws import (
    FootprintFit,
    ThroughputFit,
    fitting_footprint_law,
    fitting_throughput_law,
)


def organize_runs(sweep: pd.DataFrame) -> pd.DataFrame:
    """reading and formatting the simulation data."""

    return (
        sweep.groupby(["gpu", "atom_size", "colocation_depth"]).median().reset_index()
    )


def load_sweep(path: str) -> pd.DataFrame:
    """Function loading data for calibrations."""

    sweep_data = pd.read_csv(path)

    return sweep_data


def calibrate_throughput(medians: pd.DataFrame, gpu: str) -> ThroughputFit:
    """Fitting the throughput parameters."""

    one_gpu = medians[medians["gpu"] == gpu]
    atom_size = one_gpu["atom_size"].to_numpy()
    throughput = one_gpu["throughput"].to_numpy()

    return fitting_throughput_law(atom_size, throughput)


def calibrate_footprint(medians: pd.DataFrame, gpu: str) -> FootprintFit:
    """Fitting the footprint parameters."""

    one_gpu = medians[medians["gpu"] == gpu]
    atom_size = one_gpu["atom_size"].to_numpy()
    smact = one_gpu["sm_active"].to_numpy()

    return fitting_footprint_law(atom_size, smact)
