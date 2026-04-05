"""Load PhysioNet CinC 2019 PSV files into pandas DataFrames.

Each .psv file contains pipe-separated hourly ICU observations for one patient.
This module handles parsing individual files, assembling full training sets,
and round-tripping processed DataFrames through parquet.
"""

from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED, TRAINING_A, TRAINING_B


def load_psv(file_path: Path) -> pd.DataFrame:
    """Parse a single pipe-separated values file into a DataFrame.

    Parameters
    ----------
    file_path : Path
        Absolute or relative path to a ``.psv`` file.

    Returns
    -------
    pd.DataFrame
        Raw data with one row per hour of ICU stay.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If the file is empty or cannot be parsed.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PSV file not found: {file_path}")

    df = pd.read_csv(file_path, sep="|")
    if df.empty:
        raise ValueError(f"PSV file is empty: {file_path}")

    return df


def load_training_set(directory: Path) -> pd.DataFrame:
    """Load every PSV file in *directory* into a single DataFrame.

    A ``patient_id`` column is added, derived from each filename's stem
    (e.g. ``"p001237"`` from ``p001237.psv``).

    Progress is printed to stdout every 1 000 files.

    Parameters
    ----------
    directory : Path
        Folder containing ``.psv`` patient files.

    Returns
    -------
    pd.DataFrame
        Combined data across all patients, with ``patient_id`` column.

    Raises
    ------
    FileNotFoundError
        If *directory* does not exist.
    ValueError
        If no ``.psv`` files are found in *directory*.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Training set directory not found: {directory}")

    psv_files = sorted(directory.glob("*.psv"))
    if not psv_files:
        raise ValueError(f"No .psv files found in {directory}")

    frames: list[pd.DataFrame] = []
    total = len(psv_files)

    for idx, psv_path in enumerate(psv_files, start=1):
        patient_df = load_psv(psv_path)
        patient_df = patient_df.assign(patient_id=psv_path.stem)
        frames.append(patient_df)

        if idx % 1000 == 0:
            print(f"  Loaded {idx:,}/{total:,} files from {directory.name}")

    print(f"  Finished: {total:,} files from {directory.name}")
    return pd.concat(frames, ignore_index=True)


def load_all_data() -> pd.DataFrame:
    """Load training sets A and B and return a combined DataFrame.

    A ``hospital`` column is added (``"A"`` or ``"B"``) to indicate the
    source training set.

    Returns
    -------
    pd.DataFrame
        Combined data from both hospitals with ``patient_id`` and
        ``hospital`` columns.
    """
    print("Loading training_setA ...")
    set_a = load_training_set(TRAINING_A).assign(hospital="A")

    print("Loading training_setB ...")
    set_b = load_training_set(TRAINING_B).assign(hospital="B")

    combined = pd.concat([set_a, set_b], ignore_index=True)
    print(f"Total rows: {len(combined):,}")
    return combined


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Save a DataFrame as parquet in the processed-data directory.

    The ``.parquet`` extension is appended automatically if *name* does
    not already end with it.

    Parameters
    ----------
    df : pd.DataFrame
        Data to persist.
    name : str
        Filename (with or without ``.parquet`` extension).

    Returns
    -------
    Path
        Absolute path to the written file.

    Raises
    ------
    ValueError
        If *df* is empty or *name* is blank.
    """
    if df.empty:
        raise ValueError("Cannot save an empty DataFrame.")
    if not name or not name.strip():
        raise ValueError("File name must not be blank.")

    if not name.endswith(".parquet"):
        name = f"{name}.parquet"

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / name
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")
    return out_path


def load_processed(name: str) -> pd.DataFrame:
    """Read a parquet file from the processed-data directory.

    Parameters
    ----------
    name : str
        Filename (with or without ``.parquet`` extension).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the parquet file does not exist.
    """
    if not name.endswith(".parquet"):
        name = f"{name}.parquet"

    path = DATA_PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")

    return pd.read_parquet(path)
