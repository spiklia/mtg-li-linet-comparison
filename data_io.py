"""
data_io.py

Reading and filtering LINET / MTG-LI lightning data.
All files share one format: lon lat YYYYMMDD HH:MM:SS.SSS, whitespace-separated.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# Domain of interest (Italy and surrounding seas)
DOMAIN = {
    "lon_min": 6.0,
    "lon_max": 18.0,
    "lat_min": 37.0,
    "lat_max": 47.0,
}


def read_strokes(filepath, domain=None):
    """
    Read one lightning file (LINET or LI) and return a DataFrame with columns:
        lon, lat, datetime (np.datetime64[ns])

    Parameters
    ----------
    filepath : str or Path
        Path to the .txt file.
    domain : dict or None
        If given, keep only events inside {lon_min, lon_max, lat_min, lat_max}.
        If None, return everything.

    Returns
    -------
    pd.DataFrame with columns ['lon', 'lat', 'datetime'], sorted by time.
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "date", "time"],
        dtype={"lon": float, "lat": float, "date": str, "time": str},
        engine="c",
    )

    # Combine date and time into a single datetime column.
    # Format: YYYYMMDD HH:MM:SS.fff
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%Y%m%d %H:%M:%S.%f"
    )
    df = df.drop(columns=["date", "time"])

    if domain is not None:
        mask = (
            (df["lon"] >= domain["lon_min"])
            & (df["lon"] <= domain["lon_max"])
            & (df["lat"] >= domain["lat_min"])
            & (df["lat"] <= domain["lat_max"])
        )
        df = df.loc[mask].copy()

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def read_strokes_dir(dir_path, pattern="*.txt", domain=None, verbose=True):
    """
    Read all matching files in a directory and concatenate them into one DataFrame.

    Parameters
    ----------
    dir_path : str or Path
        Directory with the files (e.g. 'linet_data/' or 'LI_september/daily/').
    pattern : str
        Glob pattern, e.g. '*_uncorrected*.txt' for LI before parallax correction.
    domain : dict or None
        Spatial filter, see read_strokes.
    verbose : bool
        Print per-file counts.

    Returns
    -------
    pd.DataFrame with all events from all files, sorted by time.
    """
    dir_path = Path(dir_path)
    files = sorted(dir_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {dir_path}")

    dfs = []
    for f in files:
        df = read_strokes(f, domain=domain)
        if verbose:
            print(f"  {f.name}: {len(df):>7} strokes" + (" (in domain)" if domain else ""))
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    if verbose:
        print(f"Total: {len(out)} events from {len(files)} files")
    return out


def summary(df, label=""):
    """Short summary of a DataFrame, as a sanity check after loading."""
    if len(df) == 0:
        print(f"[{label}] empty")
        return
    print(f"[{label}]")
    print(f"  events:     {len(df)}")
    print(f"  time range: {df['datetime'].min()}  →  {df['datetime'].max()}")
    print(f"  lon:        {df['lon'].min():.3f}  →  {df['lon'].max():.3f}")
    print(f"  lat:        {df['lat'].min():.3f}  →  {df['lat'].max():.3f}")
    print(f"  days:       {df['datetime'].dt.date.nunique()}")


if __name__ == "__main__":
    # Example: python data_io.py <linet_dir> <li_dir>
    import sys

    linet_dir = sys.argv[1] if len(sys.argv) > 1 else "linet_data"
    li_dir = sys.argv[2] if len(sys.argv) > 2 else "LI_september/daily"

    print("=== LINET strokes ===")
    linet = read_strokes_dir(linet_dir, pattern="LinetStrokesData*.txt", domain=DOMAIN)
    summary(linet, "LINET strokes (in domain)")

    print("\n=== LI (uncorrected) ===")
    li = read_strokes_dir(li_dir, pattern="*_uncorrected*.txt", domain=DOMAIN)
    summary(li, "LI uncorrected (in domain)")
